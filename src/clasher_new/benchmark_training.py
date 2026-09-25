"""Cloud sniper throughput benchmark. No checkpoints or repository files written.

python /tmp/benchmark_training.py --source /path/to/src/clasher_new
Use the same checkpoint, software, and options on every node. Run alone in a job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def worker(source, target, rank, seed):
    def create():
        os.chdir(source)
        sys.path.insert(0, source)
        import train_sniper
        from train_legality import make_env
        train_sniper.TARGET = Path(target)
        return make_env(rank, seed, False, train_sniper.FrozenTarget)()
    return create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--rounds", type=int, default=3, help="Measured PPO rollouts per configuration")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-steps", type=int, help="Default: checkpoint's rollout length")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    target = (args.checkpoint or source / "cr_spatial_scratch/selfplay/cr_11100000_steps.zip").resolve()
    if not (source / "train_sniper.py").is_file() or not target.is_file():
        parser.error("Source must contain train_sniper.py and checkpoint must exist")
    if min(args.rounds, args.warmup, args.repeats) < 1 or (args.n_steps is not None and args.n_steps < 2):
        parser.error("rounds, warmup and repeats must be positive; n-steps must be >=2")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    allocated = min(len(affinity), int(os.environ.get("SLURM_CPUS_PER_TASK", len(affinity))))
    counts = sorted(set(args.workers or [max(1, allocated // 4), max(1, allocated // 2), allocated]))
    if min(counts) < 1:
        parser.error("worker counts must be positive")
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="cr-throughput-")) / "results.json"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(source)
    sys.path.insert(0, str(source))
    import torch
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from masked_spatial import MaskedSpatialPolicy
    torch.set_num_threads(1)

    def cpu_seconds():
        # POSIX ps avoids adding dependencies; CPU times have coarse resolution.
        table = {}
        for line in subprocess.check_output(["ps", "-axo", "pid=,ppid=,time="], text=True).splitlines():
            pid, parent, stamp = line.split()
            days = 0
            if "-" in stamp:
                day, stamp = stamp.split("-", 1)
                days = int(day)
            seconds = 0.0
            for field in stamp.split(":"):
                seconds = seconds * 60 + float(field)
            table[int(pid)] = (int(parent), seconds + days * 86400)
        family = {os.getpid()}
        while True:
            expanded = family | {pid for pid, (parent, _) in table.items() if parent in family}
            if expanded == family:
                break
            family = expanded
        return sum(table[pid][1] for pid in family if pid in table)

    class TimedPPO(PPO):
        def synchronize(self):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        def collect_rollouts(self, *a, **kw):
            self.synchronize()
            self.started = time.perf_counter()
            self.cpu_started = cpu_seconds()
            self.steps_started = self.num_timesteps
            result = super().collect_rollouts(*a, **kw)
            self.synchronize()
            self.collected = time.perf_counter()
            return result

        def train(self):
            super().train()
            self.synchronize()
            ended = time.perf_counter()
            elapsed = ended - self.started
            row = dict(steps=self.num_timesteps - self.steps_started,
                       seconds=elapsed, rollout_seconds=self.collected - self.started,
                       update_seconds=ended - self.collected,
                       average_cpu_cores=(cpu_seconds() - self.cpu_started) / elapsed,
                       ppo_updates=int(self._n_updates))
            self.samples.append(row)
            phase = "warmup" if len(self.samples) <= args.warmup else "measured"
            print(f"  {phase}: {row['steps']/elapsed:.1f} steps/s; "
                  f"rollout {row['rollout_seconds']:.1f}s, update {row['update_seconds']:.1f}s; "
                  f"CPU cores busy {row['average_cpu_cores']:.1f}", flush=True)

    digest = hashlib.sha256()
    with target.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    result = dict(host=platform.node(), platform=platform.platform(), affinity=affinity,
                  allocated_cpus=allocated, torch=torch.__version__, sb3=stable_baselines3.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  checkpoint=str(target), checkpoint_sha256=digest.hexdigest(),
                  source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                 for name in ("train_sniper.py", "environment.py", "battle.py", "masked_spatial.py")},
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, runs=[])
    print(f"Allocated CPUs: {allocated}; workers: {counts}; results: {output}", flush=True)
    for repeat in range(args.repeats):
        for count in counts:
            print(f"Workers={count}, repeat={repeat+1}", flush=True)
            started = time.perf_counter()
            env = VecMonitor(SubprocVecEnv([
                worker(str(source), str(target), rank, args.seed + repeat)
                for rank in range(count)
            ], start_method="spawn"))
            try:
                custom = dict(observation_space=env.observation_space, policy_class=MaskedSpatialPolicy)
                if args.n_steps is not None:
                    custom["n_steps"] = args.n_steps
                model = TimedPPO.load(target, env=env, device=args.device, custom_objects=custom)
                model.set_random_seed(args.seed + repeat)
                model.set_logger(configure(folder=str(output.parent), format_strings=[]))
                model.samples = []
                startup = time.perf_counter() - started
                model.learn(total_timesteps=(args.warmup + args.rounds) * count * model.n_steps,
                            reset_num_timesteps=False, log_interval=None)
                samples = model.samples[args.warmup:]
                seconds = sum(row["seconds"] for row in samples)
                steps = sum(row["steps"] for row in samples)
                run = dict(workers=count, repeat=repeat, device=str(model.device),
                           startup_seconds=startup, n_steps=model.n_steps, batch_size=model.batch_size,
                           n_epochs=model.n_epochs, samples=samples, steps_per_second=steps/seconds,
                           hours_per_million=1e6/steps*seconds/3600,
                           average_cpu_cores=sum(row["average_cpu_cores"]*row["seconds"] for row in samples)/seconds)
                result["runs"].append(run)
                output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"  RESULT: {steps/seconds:.1f} steps/s, {run['hours_per_million']:.2f} h/million", flush=True)
                del model
            finally:
                env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    ranking = sorted([(statistics.median(r["steps_per_second"] for r in result["runs"] if r["workers"] == n), n)
                      for n in counts], reverse=True)
    print("\nWorker ranking (median training steps/s):", ranking)
    print(f"Best tested: {ranking[0][1]} workers. Results: {output}")


if __name__ == "__main__":
    main()
