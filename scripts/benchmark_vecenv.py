#!/usr/bin/env python3
# flake8: noqa: T201
"""Benchmark comparing VecEnv vs sequential environment execution."""

import argparse
import time
import numpy as np
import gymnasium


def run_sequential(num_envs: int, num_steps: int, track: str = None):
    """Run environments sequentially."""
    import pystk2_gymnasium  # noqa: F401

    envs = []
    for _ in range(num_envs):
        env = gymnasium.make(
            "supertuxkart/simple-v0",
            track=track,
            num_kart=3,
            max_paths=50,
        )
        envs.append(env)

    # Warmup: reset all envs
    for env in envs:
        env.reset()

    # Warmup path cache by doing a few steps
    for env in envs:
        action = env.action_space.sample()
        env.step(action)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        for env in envs:
            action = env.action_space.sample()
            env.step(action)
    elapsed = time.perf_counter() - start

    for env in envs:
        env.close()

    return elapsed


def run_vecenv(num_envs: int, num_steps: int, track: str = None):
    """Run environments in parallel using VecEnv."""
    import pystk2_gymnasium

    vec_env = pystk2_gymnasium.make_stkrace_vec(
        num_envs=num_envs,
        env_id="supertuxkart/simple-v0",
        env_kwargs={"track": track, "num_kart": 3, "max_paths": 50},
    )

    # Sample actions in the correct format for VecEnv (dict of batched arrays)
    def sample_actions():
        # VecEnv expects {key: array of shape (num_envs, ...)}
        single_space = vec_env.single_action_space
        actions = {}
        for key, space in single_space.items():
            samples = np.array([space.sample() for _ in range(num_envs)])
            actions[key] = samples
        return actions

    # Warmup: reset
    vec_env.reset()

    # Warmup path cache by doing a few steps
    vec_env.step(sample_actions())

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        vec_env.step(sample_actions())
    elapsed = time.perf_counter() - start

    vec_env.close()

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VecEnv vs sequential environment execution"
    )
    parser.add_argument(
        "-n",
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100,
        help="Number of steps per environment (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--track",
        type=str,
        default=None,
        help="Track name (default: random)",
    )
    args = parser.parse_args()

    num_envs = args.num_envs
    num_steps = args.steps
    track = args.track

    print(f"Benchmark: {num_envs} environments, {num_steps} steps each, track={track}")
    print("=" * 60)

    # Run sequential benchmark
    print("\nRunning sequential benchmark...")
    seq_time = run_sequential(num_envs, num_steps, track)
    seq_steps_per_sec = (num_envs * num_steps) / seq_time
    print(f"Sequential: {seq_time:.2f}s ({seq_steps_per_sec:.1f} steps/sec)")

    # Run VecEnv benchmark
    print("\nRunning VecEnv benchmark...")
    vec_time = run_vecenv(num_envs, num_steps, track)
    vec_steps_per_sec = (num_envs * num_steps) / vec_time
    print(f"VecEnv:     {vec_time:.2f}s ({vec_steps_per_sec:.1f} steps/sec)")

    # Summary
    print("\n" + "=" * 60)
    speedup = seq_time / vec_time
    print(f"Speedup: {speedup:.2f}x")
    print(f"VecEnv is {'faster' if speedup > 1 else 'slower'} than sequential")


if __name__ == "__main__":
    main()
