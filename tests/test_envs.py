import gymnasium as gym
import pytest
import pystk2_gymnasium  # noqa: F401
from pystk2_gymnasium.envs import AgentSpec

envs = [key for key in gym.envs.registry.keys() if key.startswith("supertuxkart/")]


@pytest.mark.parametrize("name", envs)
@pytest.mark.parametrize("use_ai", [True, False])
def test_env(name, use_ai):
    env = None
    if name.startswith("supertuxkart/multi-"):
        kwargs = {"agents": [AgentSpec(use_ai=use_ai), AgentSpec(use_ai=use_ai)]}
    else:
        kwargs = {"agent": AgentSpec(use_ai=use_ai)}
    try:
        env = gym.make(name, render_mode=None, **kwargs)

        ix = 0
        done = False
        state, *_ = env.reset()

        for _ in range(10):
            ix += 1
            action = env.action_space.sample()
            # print(action)
            state, reward, terminated, truncated, _ = env.step(action)
            done = truncated or terminated
            if done:
                break
    finally:
        if env is not None:
            env.close()
