"""Tests for vectorized environment support."""

import numpy as np
import pytest

import pystk2_gymnasium  # noqa: F401


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize(
    "env_id",
    ["supertuxkart/simple-v0", "supertuxkart/full-v0"],
)
def test_vecenv_creation(num_envs, env_id):
    """Test that VecEnv can be created with different numbers of environments."""
    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id=env_id,
            env_kwargs={"num_kart": 2},
        )
        assert vec_env.num_envs == num_envs
    finally:
        if vec_env is not None:
            vec_env.close()


@pytest.mark.parametrize("num_envs", [1, 2])
def test_vecenv_reset(num_envs):
    """Test that VecEnv reset returns correct shapes."""
    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id="supertuxkart/simple-v0",
            env_kwargs={"num_kart": 2},
        )

        observations, infos = vec_env.reset()

        # Check that observations is a dict with batched values
        assert isinstance(observations, dict)
        # Check a few observation keys have the right batch dimension
        assert observations["phase"].shape[0] == num_envs
        assert observations["energy"].shape[0] == num_envs
    finally:
        if vec_env is not None:
            vec_env.close()


@pytest.mark.parametrize("num_envs", [1, 2])
def test_vecenv_step(num_envs):
    """Test that VecEnv step returns correct shapes."""
    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id="supertuxkart/simple-v0",
            env_kwargs={"num_kart": 2},
        )

        vec_env.reset()

        # Sample actions for all environments
        actions = vec_env.action_space.sample()
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)

        # Check shapes
        assert isinstance(observations, dict)
        assert observations["phase"].shape[0] == num_envs
        assert rewards.shape == (num_envs,)
        assert terminations.shape == (num_envs,)
        assert truncations.shape == (num_envs,)
    finally:
        if vec_env is not None:
            vec_env.close()


def test_vecenv_multiple_steps():
    """Test running multiple steps in VecEnv."""
    num_envs = 2
    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id="supertuxkart/simple-v0",
            env_kwargs={"num_kart": 2},
        )

        vec_env.reset()

        for _ in range(5):
            actions = vec_env.action_space.sample()
            observations, rewards, terminations, truncations, infos = vec_env.step(
                actions
            )

            # Basic sanity checks
            assert observations["phase"].shape[0] == num_envs
            assert np.isfinite(rewards).all()
    finally:
        if vec_env is not None:
            vec_env.close()


def test_vecenv_shared_memory_auto_detection():
    """Test that shared_memory is auto-detected based on observation space."""
    from pystk2_gymnasium.vecenv import _has_dynamic_spaces
    import gymnasium as gym

    # simple-v0 has ConstantSizedObservations, should NOT have dynamic spaces
    env_simple = gym.make("supertuxkart/simple-v0", num_kart=2)
    try:
        assert not _has_dynamic_spaces(env_simple.observation_space)
    finally:
        env_simple.close()

    # full-v0 has Sequence spaces, should have dynamic spaces
    env_full = gym.make("supertuxkart/full-v0", num_kart=2)
    try:
        assert _has_dynamic_spaces(env_full.observation_space)
    finally:
        env_full.close()


# --- Multi-agent VecEnv tests ---


@pytest.mark.parametrize("num_envs", [1, 2])
def test_vecenv_multi_agent_creation(num_envs):
    """Test that multi-agent VecEnv can be created."""
    from pystk2_gymnasium import AgentSpec

    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id="supertuxkart/multi-full-v0",
            env_kwargs={
                "num_kart": 3,
                "agents": [AgentSpec(), AgentSpec()],
            },
        )
        assert vec_env.num_envs == num_envs
    finally:
        if vec_env is not None:
            vec_env.close()


def test_vecenv_multi_agent_reset():
    """Test that multi-agent VecEnv reset returns correct structure."""
    from pystk2_gymnasium import AgentSpec

    num_envs = 2
    num_agents = 2
    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id="supertuxkart/multi-full-v0",
            env_kwargs={
                "num_kart": 3,
                "agents": [AgentSpec() for _ in range(num_agents)],
            },
        )

        observations, infos = vec_env.reset()

        # Multi-agent: observations is a dict keyed by agent index
        assert isinstance(observations, dict)
        # Should have keys for each agent ("0", "1")
        assert "0" in observations
        assert "1" in observations

        # Each agent's observations should be batched across environments
        assert observations["0"]["phase"].shape[0] == num_envs
        assert observations["1"]["phase"].shape[0] == num_envs
    finally:
        if vec_env is not None:
            vec_env.close()


def test_vecenv_multi_agent_step():
    """Test that multi-agent VecEnv step works correctly."""
    from pystk2_gymnasium import AgentSpec

    num_envs = 2
    num_agents = 2
    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id="supertuxkart/multi-full-v0",
            env_kwargs={
                "num_kart": 3,
                "agents": [AgentSpec() for _ in range(num_agents)],
            },
        )

        vec_env.reset()

        # Sample actions for all environments
        actions = vec_env.action_space.sample()
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)

        # Check observation structure
        assert isinstance(observations, dict)
        assert "0" in observations
        assert "1" in observations
        assert observations["0"]["phase"].shape[0] == num_envs

        # Check rewards, terminations, truncations shapes
        assert rewards.shape == (num_envs,)
        assert terminations.shape == (num_envs,)
        assert truncations.shape == (num_envs,)
    finally:
        if vec_env is not None:
            vec_env.close()


def test_vecenv_multi_agent_multiple_steps():
    """Test running multiple steps in multi-agent VecEnv."""
    from pystk2_gymnasium import AgentSpec

    num_envs = 2
    num_agents = 2
    vec_env = None
    try:
        vec_env = pystk2_gymnasium.make_stkrace_vec(
            num_envs=num_envs,
            env_id="supertuxkart/multi-full-v0",
            env_kwargs={
                "num_kart": 3,
                "agents": [AgentSpec() for _ in range(num_agents)],
            },
        )

        vec_env.reset()

        for _ in range(5):
            actions = vec_env.action_space.sample()
            observations, rewards, terminations, truncations, infos = vec_env.step(
                actions
            )

            # Basic sanity checks
            assert observations["0"]["phase"].shape[0] == num_envs
            assert observations["1"]["phase"].shape[0] == num_envs
            assert np.isfinite(rewards).all()
    finally:
        if vec_env is not None:
            vec_env.close()
