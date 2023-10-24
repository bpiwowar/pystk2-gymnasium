from gymnasium import spaces
import gymnasium as gym
import numpy as np


class SpaceFlattener:
    def flatten_space(self, space: gym.Space):
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
            return discrete_space

        return spaces.Dict(
            {
                "env_obs/discrete": discrete_space,
                "env_obs/continuous": spaces.Box(
                    low=np.concatenate(lows),
                    high=np.concatenate(highs),
                    shape=(continuous_size,),
                    dtype=np.float32,
                ),
            }
        )


class ObsFlattenerWrapper(gym.ObservationWrapper, SpaceFlattener):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = self.flatten_space(env.observation_space)

        # Adds action in the space
        self.has_action = env.observation_space.get("action", None) is not None
        if self.has_action:
            self.observation_space["action"] = env.action_space

    def observation(self, observation):
        action = {}
        if self.has_action:
            action_dict = {
                key: np.array([value]) for key, value in observation["action"].items()
            }
            from .envs import DiscreteActionSTKRaceEnv

            if isinstance(self.unwrapped, DiscreteActionSTKRaceEnv):
                action_dict = self.unwrapped.to_discrete(action_dict)
            action = {f"action/{key}": value for key, value in action_dict.items()}

        return {
            **action,
            "env_obs/discrete": np.array(
                [observation[key] for key in self.discrete_keys]
            ),
            "env_obs/continuous": np.concatenate(
                [observation[key].flatten() for key in self.continuous_keys]
            ),
        }


class ActionFlattenerWrapper(gym.ActionWrapper, SpaceFlattener):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = self.flatten_space(env.action_space)

    def action(self, action):
        if self.only_discrete:
            assert len(self.discrete_keys) == len(action), (
                "Not enough discrete values: "
                f"""expected {len(self.discrete_keys)}, got {len(action)}"""
            )
            return {
                key: key_action for key, key_action in zip(self.discrete_keys, action)
            }

        assert len(self.discrete_keys) == len(
            action["env_obs/discrete"]
        ), "Not enough discrete values: "
        f"""expected {len(self.discrete_keys)}, got {len(action["env_obs/discrete"])}"""
        discrete = {
            key: key_action
            for key, key_action in zip(self.discrete_keys, action["env_obs/discrete"])
        }
        continuous = {
            key: action["env_obs/continuous"][
                self.indices[ix] : self.indices[ix + 1]
            ].reshape(shape)
            for ix, (key, shape) in enumerate(zip(self.continuous_keys, self.shapes))
        }
        return {**discrete, **continuous}
