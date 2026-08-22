"""Benchmark batched model decisions against the random opponent.

Run from this directory, for example:

    PYTHONPATH=. python benchmark_speed.py --n-envs 16 --seconds 60 --device cuda

Pass ``--model path/to/checkpoint.zip`` when a trained checkpoint is available.

"speed" here means wall-clock speed only.  Each RL decision still advances the
normal 30 simulator frames (0.5 seconds); visualization is disabled.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment import CREnv, legal_random_strategy
from train import CRPolicy, CRSetExtractor


def make_env(rank: int, speed: float):
    def factory():
        # Keep worker RNG streams distinct while leaving simulator timing unchanged.
        seed = 100_000 + rank
        np.random.seed(seed)
        return CREnv(opponent_model=legal_random_strategy, visualize=False, speed=speed)

    return factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="checkpoint to benchmark; omit to benchmark a fresh untrained policy")
    parser.add_argument("--n-envs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=100,
                        help="warmup batched decisions before timing")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="simulator timing multiplier; keep at 1 for normal gameplay")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--deterministic", action="store_true",
                        help="benchmark deterministic inference instead of sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_envs < 1 or args.seconds <= 0 or args.warmup < 0:
        raise ValueError("--n-envs and --seconds must be positive; --warmup cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    start_method = "spawn"
    env = VecMonitor(SubprocVecEnv(
        [make_env(rank, args.speed) for rank in range(args.n_envs)],
        start_method=start_method,
    ))
    if args.model:
        model = PPO.load(args.model, env=env, device=args.device)
    else:
        model = PPO(
            CRPolicy,
            env,
            policy_kwargs={"features_extractor_class": CRSetExtractor,
                           "net_arch": {"pi": [], "vf": []}},
            n_steps=8,
            batch_size=8,
            device=args.device,
            verbose=0,
        )
    obs = env.reset()

    # Warm up subprocesses, PyTorch kernels, and CUDA context before timing.
    for _ in range(args.warmup):
        actions, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, _, _ = env.step(actions)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        torch.cuda.synchronize()

    policy_seconds = 0.0
    simulator_seconds = 0.0
    decisions = 0
    episodes = 0
    simulated_seconds = 0.0
    deadline = time.perf_counter() + args.seconds
    wall_start = time.perf_counter()

    while time.perf_counter() < deadline:
        policy_start = time.perf_counter()
        actions, _ = model.predict(obs, deterministic=args.deterministic)
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
            torch.cuda.synchronize()
        policy_seconds += time.perf_counter() - policy_start

        step_start = time.perf_counter()
        obs, _, dones, infos = env.step(actions)
        simulator_seconds += time.perf_counter() - step_start
        decisions += args.n_envs
        simulated_seconds += args.n_envs * 0.5 * args.speed
        episodes += int(np.asarray(dones).sum())

    wall_seconds = time.perf_counter() - wall_start
    env.close()

    total_decisions_per_second = decisions / wall_seconds
    print(f"model:                         {args.model}")
    print(f"device:                        {args.device}")
    print(f"environments:                  {args.n_envs}")
    print(f"simulator speed multiplier:    {args.speed}")
    print(f"timed wall seconds:            {wall_seconds:.3f}")
    print(f"completed games:               {episodes}")
    print(f"simulated game-seconds:        {simulated_seconds:.1f}")
    print(f"aggregate decisions/sec:       {total_decisions_per_second:.2f}")
    print(f"decisions/sec/environment:     {total_decisions_per_second / args.n_envs:.2f}")
    print(f"simulated game-seconds/wall-s: {simulated_seconds / wall_seconds:.2f}")
    print(f"policy inference wall time:    {policy_seconds:.3f}s")
    print(f"simulator step wall time:      {simulator_seconds:.3f}s")
    print(f"policy time fraction:           {policy_seconds / wall_seconds:.1%}")
    print(f"simulator time fraction:        {simulator_seconds / wall_seconds:.1%}")


if __name__ == "__main__":
    main()
