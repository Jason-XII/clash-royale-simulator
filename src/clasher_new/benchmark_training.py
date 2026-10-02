"""Benchmark parallel PPO throughput on one cloud node.

This benchmark writes only its JSON result file and temporary worker buffers.
Run the same command on each node with the same checkpoint and software:

    python benchmark_training.py --source /path/to/src/clasher_new \
        --workers 4 8 16 24 --rounds 3 --warmup 1 --repeats 2 \
        --device cuda

A checkpoint is optional. Without one, every run creates the same fresh
MaskedSpatialPolicy and uses scripted opponents. With one, the benchmark loads
that policy and uses the checkpoint as a frozen opponent too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

# Keep BLAS and Torch from creating hidden thread pools in every worker.
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def process_cpu_seconds():
    """Return CPU seconds used by this process and its live descendants."""
    try:
        rows = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,time="], text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return time.process_time()
    table = {}
    for row in rows:
        try:
            pid, parent, stamp = row.split()
            seconds = 0.0
            if "-" in stamp:
                days, stamp = stamp.split("-", 1)
                seconds += int(days) * 86400
            for field in stamp.split(":"):
                seconds = seconds * 60 + float(field)
            table[int(pid)] = (int(parent), seconds)
        except (ValueError, TypeError):
            continue
    family = {os.getpid()}
    while True:
        children = {pid for pid, (parent, _) in table.items() if parent in family}
        expanded = family | children
        if expanded == family:
            break
        family = expanded
    return sum(table[pid][1] for pid in family if pid in table)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker_factory(source, target, rank, seed):
    """Create one active training environment in a spawned worker."""
    def create():
        os.chdir(source)
        sys.path.insert(0, source)
        from train_core import OpponentFactory, make_env
        opponent_factory = (
            OpponentFactory(forced=str(target)) if target
            else OpponentFactory()
        )
        return make_env(
            rank=rank,
            seed=seed,
            opponent_factory=opponent_factory,
        )()
    return create


def available_cpus():
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    else:
        affinity = list(range(os.cpu_count() or 1))
    requested = int(os.environ.get("SLURM_CPUS_PER_TASK", len(affinity)))
    return affinity, max(1, min(len(affinity), requested))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint shared by all node runs.")
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--max-workers", type=int, default=32,
                        help="Default worker sweep is capped here to avoid OOM.")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Measured PPO rollouts per configuration.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-steps", type=int,
                        help="Override the rollout length; default is 512 for a fresh model.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    source = args.source.resolve()
    target = args.checkpoint.resolve() if args.checkpoint else None
    if target is not None and not target.is_file():
        raise SystemExit(f"checkpoint does not exist: {target}")
    if not (source / "train_core.py").is_file():
        raise SystemExit(f"source does not contain train_core.py: {source}")
    if min(args.rounds, args.warmup, args.repeats) < 1:
        raise SystemExit("rounds, warmup, and repeats must be positive")
    if args.n_steps is not None and args.n_steps < 2:
        raise SystemExit("n-steps must be at least 2")
    if args.max_workers < 1:
        raise SystemExit("max-workers must be positive")

    affinity, allocated = available_cpus()
    default_counts = [max(1, allocated // 4), max(1, allocated // 2),
                      min(allocated, args.max_workers)]
    counts = sorted(set(args.workers or default_counts))
    if min(counts) < 1:
        raise SystemExit("worker counts must be positive")
    if max(counts) > allocated:
        print(f"warning: requested {max(counts)} workers but only {allocated} CPUs are allocated",
              flush=True)

    output = (args.output.resolve() if args.output else
              Path(tempfile.mkdtemp(prefix="cr-throughput-")) / "results.json")
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(source)
    sys.path.insert(0, str(source))

    import stable_baselines3
    import torch
    from masked_spatial import MaskedSpatialPolicy
    from parallel_rollout import ParallelPPO, ParallelVecEnv, VariableDiscountBuffer
    from train_core import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import get_device

    torch.set_num_threads(1)
    resolved_device = str(get_device(args.device))
    print(f"requested_device={args.device} resolved_device={resolved_device}", flush=True)
    if (args.device == "auto" and resolved_device == "cpu"
            and torch.backends.mps.is_available()):
        print("note: auto selected CPU; use --device mps on this Mac", flush=True)

    class TimedPPO(ParallelPPO):
        def synchronize(self):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elif self.device.type == "mps":
                torch.mps.synchronize()

        def collect_rollouts(self, *call_args, **call_kwargs):
            self.synchronize()
            self.phase_started = time.perf_counter()
            self.cpu_started = process_cpu_seconds()
            self.steps_started = self.num_timesteps
            result = super().collect_rollouts(*call_args, **call_kwargs)
            self.synchronize()
            self.rollout_finished = time.perf_counter()
            self.cpu_rollout_finished = process_cpu_seconds()
            return result

        def train(self):
            super().train()
            self.synchronize()
            finished = time.perf_counter()
            elapsed = finished - self.phase_started
            rollout_seconds = self.rollout_finished - self.phase_started
            update_seconds = finished - self.rollout_finished
            cpu_total = process_cpu_seconds() - self.cpu_started
            cpu_rollout = self.cpu_rollout_finished - self.cpu_started
            steps = self.num_timesteps - self.steps_started
            self.samples.append({
                "steps": steps,
                "seconds": elapsed,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "rollout_steps_per_second": steps / rollout_seconds,
                "training_steps_per_second": steps / elapsed,
                "rollout_cpu_cores": cpu_rollout / rollout_seconds,
                "average_cpu_cores": cpu_total / elapsed,
                "ppo_updates": int(self._n_updates),
            })
            phase = "warmup" if len(self.samples) <= args.warmup else "measured"
            row = self.samples[-1]
            print(
                f"  {phase}: {row['training_steps_per_second']:.1f} steps/s "
                f"(rollout {row['rollout_steps_per_second']:.1f}); "
                f"rollout {row['rollout_seconds']:.1f}s, "
                f"update {row['update_seconds']:.1f}s, "
                f"CPU {row['average_cpu_cores']:.1f} cores",
                flush=True,
            )

    source_files = (
        "train_core.py", "environment.py", "battle.py", "core.py",
        "player.py", "card_utils.py", "card_mechanics.py", "masked_spatial.py",
        "parallel_rollout.py", "strategies.py", "defensive_strategy.py",
    )
    result = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "affinity": affinity,
        "allocated_cpus": allocated,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "device": resolved_device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(target) if target else None,
        "checkpoint_sha256": sha256(target) if target else None,
        "opponent_mode": "frozen_checkpoint" if target else "scripted",
        "source_sha256": {name: sha256(source / name) for name in source_files
                          if (source / name).is_file()},
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "runs": [],
    }
    print(
        f"host={result['host']} allocated_cpus={allocated} workers={counts}\n"
        f"checkpoint={target}\nresults={output}",
        flush=True,
    )

    order = [(repeat, count) for repeat in range(args.repeats) for count in counts]
    random.Random(args.seed).shuffle(order)
    for repeat, count in order:
        print(f"Workers={count}, repeat={repeat + 1}", flush=True)
        started = time.perf_counter()
        env = None
        try:
            env = ParallelVecEnv([
                worker_factory(str(source), str(target) if target else None,
                               rank, args.seed + repeat)
                for rank in range(count)
            ])
            if target:
                custom = {
                    "observation_space": env.observation_space,
                    "policy_class": MaskedSpatialPolicy,
                }
                if args.n_steps is not None:
                    custom["n_steps"] = args.n_steps
                    custom["batch_size"] = min(args.n_steps * count, 256)
                model = TimedPPO.load(
                    target, env=env, device=resolved_device, custom_objects=custom,
                )
            else:
                model = TimedPPO(
                    MaskedSpatialPolicy,
                    env,
                    n_steps=args.n_steps or 512,
                    batch_size=min(256, (args.n_steps or 512) * count),
                    learning_rate=1e-4,
                    n_epochs=4,
                    target_kl=0.03,
                    ent_coef=0.005,
                    gamma=DEFAULT_GAMMA,
                    gae_lambda=DEFAULT_GAE_LAMBDA,
                    rollout_buffer_class=VariableDiscountBuffer,
                    device=resolved_device,
                    seed=args.seed + repeat,
                    verbose=0,
                )
            model.set_random_seed(args.seed + repeat)
            model.set_logger(configure(format_strings=[]))
            model.samples = []
            startup = time.perf_counter() - started
            total_steps = (args.warmup + args.rounds) * count * model.n_steps
            model.learn(total_timesteps=total_steps, reset_num_timesteps=False,
                        log_interval=None)
            samples = model.samples[args.warmup:]
            seconds = sum(row["seconds"] for row in samples)
            rollout_seconds = sum(row["rollout_seconds"] for row in samples)
            steps = sum(row["steps"] for row in samples)
            run = {
                "workers": count,
                "repeat": repeat,
                "device": str(model.device),
                "opponent_mode": "frozen_checkpoint" if target else "scripted",
                "startup_seconds": startup,
                "n_steps": model.n_steps,
                "batch_size": model.batch_size,
                "n_epochs": model.n_epochs,
                "samples": samples,
                "steps": steps,
                "seconds": seconds,
                "rollout_seconds": rollout_seconds,
                "update_seconds": seconds - rollout_seconds,
                "steps_per_second": steps / seconds,
                "rollout_steps_per_second": steps / rollout_seconds,
                "hours_per_million": 1e6 / steps * seconds / 3600,
                "average_cpu_cores": sum(
                    row["average_cpu_cores"] * row["seconds"] for row in samples
                ) / seconds,
            }
            result["runs"].append(run)
            print(
                f"  RESULT: {run['steps_per_second']:.1f} training steps/s, "
                f"{run['rollout_steps_per_second']:.1f} rollout steps/s, "
                f"{run['hours_per_million']:.2f} h/million",
                flush=True,
            )
            del model
        except Exception as error:
            failure = {"workers": count, "repeat": repeat,
                       "error": f"{type(error).__name__}: {error}"}
            result["runs"].append(failure)
            print(f"  FAILED: {failure['error']}", flush=True)
        finally:
            if env is not None:
                env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            output.write_text(json.dumps(result, indent=2) + "\n")

    successful = [run for run in result["runs"] if "steps_per_second" in run]
    if not successful:
        raise SystemExit(f"all benchmark configurations failed; inspect {output}")
    ranking = []
    for count in counts:
        values = [run["steps_per_second"] for run in successful
                  if run["workers"] == count]
        if values:
            ranking.append({"workers": count, "median_steps_per_second": statistics.median(values),
                            "runs": len(values)})
    ranking.sort(key=lambda row: row["median_steps_per_second"], reverse=True)
    result["ranking"] = ranking
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("\nWorker ranking (median end-to-end steps/s):", ranking)
    print(f"Best tested: {ranking[0]['workers']} workers. Results: {output}")


if __name__ == "__main__":
    main()
