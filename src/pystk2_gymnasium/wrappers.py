from typing import Any
from gymnasium import spaces
import gymnasium as gym
import numpy as np


class SpaceFlattener:
    def __init__(self, space: gym.Space):
        # Flatten the observation space
        self.continuous_keys = []
        self.shapes = []
        self.discrete_keys = []
        self.indices = [0]

        continuous_size = 0
        lows = []
        highs = []
        counts = []

        for key, value in space.items():
            # Ignore the AI action
            if key == "action":
                continue

            if isinstance(value, spaces.Discrete):
                self.discrete_keys.append(key)
                counts.append(value.n)
            elif isinstance(value, spaces.Box):
                self.continuous_keys.append(key)
                self.shapes.append(value.shape)
                lows.append(value.low.flatten())
                highs.append(value.high.flatten())
                continuous_size += np.prod(value.shape)
                self.indices.append(continuous_size)
            else:
                assert False, f"Type not handled {type(value)}"

        self.only_discrete = len(lows) == 0
        discrete_space = spaces.MultiDiscrete(counts, dtype=np.int64)
        if self.only_discrete:
            self.space = discrete_space
        else:
            self.space = spaces.Dict(
                {
                    "discrete": discrete_space,
                    "continuous": spaces.Box(
                        low=np.concatenate(lows),
                        high=np.concatenate(highs),
                        shape=(continuous_size,),
                        dtype=np.float32,
                    ),
                }
            )


class FlattenerWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_flattener = SpaceFlattener(env.observation_space)
        self.observation_space = self.observation_flattener.space

        self.action_flattener = SpaceFlattener(env.action_space)
        self.action_space = self.action_flattener.space

        # Adds action in the space
        self.has_action = env.observation_space.get("action", None) is not None
        if self.has_action:
            self.observation_space["action"] = self.action_flattener.space

    def observation(self, observation):
        new_obs = {
            "discrete": np.array(
                [observation[key] for key in self.observation_flattener.discrete_keys]
            ),
            "continuous": np.concatenate(
                [
                    observation[key].flatten()
                    for key in self.observation_flattener.continuous_keys
                ]
            ),
        }

        if self.has_action:
            obs_action = observation["action"]
            discrete = np.array(
                [obs_action[key] for key in self.action_flattener.discrete_keys]
            )
            if self.action_flattener.only_discrete:
                new_obs["action"] = discrete
            else:
                continuous = np.concatenate(
                    [
                        obs_action[key].flatten()
                        for key in self.action_flattener.continuous_keys
                    ]
                )
                new_obs["action"] = {"discrete": discrete, "continuous": continuous}

        return new_obs

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return super().step(self.action(action))

    def action(self, action):
        if self.action_flattener.only_discrete:
            assert len(self.action_flattener.discrete_keys) == len(action), (
                "Not enough discrete values: "
                f"""expected {len(self.action_flattener.discrete_keys)}, """
                f"""got {len(action)}"""
            )
            action = {
                key: key_action
                for key, key_action in zip(self.action_flattener.discrete_keys, action)
            }

        else:
            assert len(self.action_flattener.discrete_keys) == len(
                action["discrete"]
            ), "Not enough discrete values: "
            f"""expected {len(self.discrete_keys)}, got {len(action["discrete"])}"""
            discrete = {
                key: key_action
                for key, key_action in zip(
                    self.action_flattener.discrete_keys, action["discrete"]
                )
            }
            continuous = {
                key: action["continuous"][
                    self.indices[ix] : self.indices[ix + 1]
                ].reshape(shape)
                for ix, (key, shape) in enumerate(
                    zip(self.action_flattener.continuous_keys, self.shapes)
                )
            }
            action = {**discrete, **continuous}

        return action
