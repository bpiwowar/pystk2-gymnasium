import gymnasium as gym
import pytest
import pystk2_gymnasium

envs = [key for key in gym.envs.registry.keys() if key.startswith("supertuxkart/")]

@pytest.mark.parametrize("name", envs)
def test_env(name):
    env = None
    try:
        env = gym.make(name, render_mode=None, use_ai=False)

        ix = 0
        done = False
        state, *_ = env.reset()

        while not done:
            ix += 1
            action = env.action_space.sample()
            # print(action)
            state, reward, terminated, truncated, _ = env.step(action)
            done = truncated or terminated        
    finally:        
        if env is not None:
            env.close()
