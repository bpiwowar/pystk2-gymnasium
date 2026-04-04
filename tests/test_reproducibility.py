"""Tests for seed-based reproducibility."""

import gymnasium as gym
import numpy as np
import pytest

import pystk2_gymnasium  # noqa: F401
from pystk2_gymnasium.envs import AgentSpec


SEED = 42
NUM_STEPS = 10
TRACK = "lighthouse"


def collect_obs_list(obs):
    """Flatten an observation dict into a list of arrays for comparison."""
    if isinstance(obs, dict):
        return {k: np.asarray(v) for k, v in obs.items()}
    return np.asarray(obs)


def assert_obs_equal(obs1, obs2, step, prefix=""):
    """Assert two observation dicts are equal."""
    assert type(obs1) is type(obs2), f"Step {step}{prefix}: type mismatch"
    if isinstance(obs1, dict):
        assert obs1.keys() == obs2.keys(), (
            f"Step {step}{prefix}: key mismatch {obs1.keys()} vs {obs2.keys()}"
        )
        for k in obs1:
            assert_obs_equal(obs1[k], obs2[k], step, prefix=f"{prefix}.{k}")
    else:
        a1, a2 = np.asarray(obs1), np.asarray(obs2)
        if a1.dtype == object:
            # Nested structure (e.g. dict inside array), skip
            return
        np.testing.assert_array_equal(
            a1, a2, err_msg=f"Step {step}{prefix}: values differ"
        )


def run_single_env(seed, num_steps, use_ai=True):
    """Run a single-agent env and collect trajectory."""
    env = gym.make(
        "supertuxkart/simple-v0",
        render_mode=None,
        agent=AgentSpec(use_ai=use_ai),
        track=TRACK,
        num_kart=3,
    )
    try:
        obs, info = env.reset(seed=seed)
        observations = [collect_obs_list(obs)]
        rewards = []

        rng = np.random.RandomState(seed + 1000)

        for _ in range(num_steps):
            if use_ai:
                action = env.action_space.sample()
            else:
                # Use a seeded RNG for actions
                action = env.action_space.seed(int(rng.randint(2**31)))
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(collect_obs_list(obs))
            rewards.append(reward)
            if terminated or truncated:
                break
        return observations, rewards
    finally:
        env.close()


def test_single_env_reproducibility():
    """Two runs of the same single-agent env with same seed produce
    identical trajectories (using AI control so actions are deterministic)."""
    obs1, rew1 = run_single_env(SEED, NUM_STEPS, use_ai=True)
    obs2, rew2 = run_single_env(SEED, NUM_STEPS, use_ai=True)

    assert len(obs1) == len(obs2), "Different trajectory lengths"
    for i, (o1, o2) in enumerate(zip(obs1, obs2)):
        assert_obs_equal(o1, o2, i)
    np.testing.assert_array_equal(rew1, rew2, err_msg="Rewards differ")


def run_vec_env(seed, num_envs, num_steps):
    """Run a vectorized env and collect trajectory."""
    vec_env = pystk2_gymnasium.make_stkrace_vec(
        num_envs=num_envs,
        env_id="supertuxkart/simple-v0",
        env_kwargs={
            "track": TRACK,
            "num_kart": 3,
            "agent": AgentSpec(use_ai=True),
        },
    )
    try:
        observations, infos = vec_env.reset(seed=seed)
        all_obs = [collect_obs_list(observations)]
        all_rewards = []

        for _ in range(num_steps):
            actions = vec_env.action_space.sample()
            observations, rewards, terminations, truncations, infos = vec_env.step(
                actions
            )
            all_obs.append(collect_obs_list(observations))
            all_rewards.append(rewards.copy())
        return all_obs, all_rewards
    finally:
        vec_env.close()


@pytest.mark.parametrize("num_envs", [1, 2])
def test_vec_env_reproducibility(num_envs):
    """Two runs of the same vec env with same seed produce identical
    trajectories (using AI control)."""
    obs1, rew1 = run_vec_env(SEED, num_envs, NUM_STEPS)
    obs2, rew2 = run_vec_env(SEED, num_envs, NUM_STEPS)

    assert len(obs1) == len(obs2), "Different trajectory lengths"
    for i, (o1, o2) in enumerate(zip(obs1, obs2)):
        assert_obs_equal(o1, o2, i)
    for i, (r1, r2) in enumerate(zip(rew1, rew2)):
        np.testing.assert_array_equal(r1, r2, err_msg=f"Rewards differ at step {i}")
