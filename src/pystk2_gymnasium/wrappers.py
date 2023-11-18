"""
This module contains generic wrappers
"""
from typing import Any, Callable, Dict, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import (
    Wrapper,
    WrapperActType,
    WrapperObsType,
    ObsType,
    ActType,
    Env,
)
import numpy as np

from pystk2_gymnasium.definitions import ActionObservationWrapper


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
        self.only_continuous = len(counts) == 0
        discrete_space = spaces.MultiDiscrete(counts, dtype=np.int64)
        if len(lows) > 0:
            continuous_space = spaces.Box(
                low=np.concatenate(lows),
                high=np.concatenate(highs),
                shape=(continuous_size,),
                dtype=np.float32,
            )

        if self.only_discrete:
            self.space = discrete_space
        elif self.only_continuous:
            self.space = continuous_space
        else:
            self.space = spaces.Dict(
                {
                    "discrete": discrete_space,
                    "continuous": continuous_space,
                }
            )


class FlattenerWrapper(ActionObservationWrapper):
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
            # Transforms from nested action to a flattened
            obs_action = observation["action"]
            discrete = np.array(
                [obs_action[key] for key in self.action_flattener.discrete_keys]
            )
            if self.action_flattener.only_discrete:
                new_obs["action"] = discrete
            else:
                continuous = np.concatenate(
                    [
                        np.array([obs_action[key]])
                        if isinstance(obs_action[key], float)
                        else obs_action[key].flatten()
                        for key in self.action_flattener.continuous_keys
                    ]
                )
                if self.action_flattener.only_continuous:
                    return continuous
                new_obs["action"] = {"discrete": discrete, "continuous": continuous}

        return new_obs

    def action(self, action):
        discrete_actions = {}
        if not self.action_flattener.only_continuous:
            actions = (
                action if self.action_flattener.only_discrete else action["discrete"]
            )
            assert len(self.action_flattener.discrete_keys) == len(actions), (
                "Not enough discrete values: "
                f"""expected {len(self.action_flattener.discrete_keys)}, """
                f"""got {len(action)}"""
            )
            discrete_actions = {
                key: key_action
                for key, key_action in zip(self.action_flattener.discrete_keys, actions)
            }

        continuous_actions = {}
        if not self.action_flattener.only_discrete:
            actions = (
                action
                if self.action_flattener.only_continuous
                else action["continuous"]
            )
            continuous_actions = {
                key: actions[
                    self.action_flattener.indices[ix] : self.action_flattener.indices[
                        ix + 1
                    ]
                ].reshape(shape)
                for ix, (key, shape) in enumerate(
                    zip(
                        self.action_flattener.continuous_keys,
                        self.action_flattener.shapes,
                    )
                )
            }

        return {**discrete_actions, **continuous_actions}


class FlattenMultiDiscreteActions(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        assert isinstance(self.action_space, spaces.MultiDiscrete)

        self.nvec = self.action_space.nvec
        self.action_space = spaces.Discrete(np.prod(self.action_space.nvec))

    def action(self, action):
        actions = []
        for n in self.nvec:
            actions.append(action % n)
            action = action // n
        return actions


class MultiMonoEnv(gym.Env):
    """Fake mono-kart environment for mono-kart wrappers"""

    def __init__(self, env: gym.Env, key: str):
        self._env = env
        self.observation_space = env.observation_space[key]
        self.action_space = env.action_space[key]

    def reset(self, **kwargs):
        raise RuntimeError("Should not be called - fake mono environment")

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        raise RuntimeError("Should not be called - fake mono environment")


class MonoAgentWrapperAdapter(ActionObservationWrapper):
    """Adapts a mono agent wrapper for a multi-agent one"""

    def __init__(
        self,
        env: gym.Env,
        *,
        wrapper_factories: Dict[str, Callable[[gym.Env], Wrapper]],
    ):
        """Initialize an adapter that use distinct wrappers

        It supposes that the space/action space is a dictionary where each key
        corresponds to a different agent.

        :param env: The base environment
        :param wrapper_factories: Return a wrapper for every key in the
            observation/action spaces dictionary. Supported wrappers are
            `ActionObservationWrapper`, `ObservationWrapper`, and `ActionWrapper`.
        """
        super().__init__(env)

        # Perform some checks
        self.keys = set(self.action_space.keys())
        assert self.keys == set(
            self.observation_space.keys()
        ), "Observation and action keys differ"

        # Setup the wrapped environment
        self.mono_envs = {}
        self.wrappers = {}

        for key in env.observation_space.keys():
            mono_env = MultiMonoEnv(env, key)
            self.mono_envs[key] = mono_env
            wrapper = wrapper_factories[key](mono_env)

            # Build up the list of action/observation wrappers
            self.wrappers[key] = wrappers = []
            while wrapper is not mono_env:
                assert isinstance(
                    wrapper,
                    (
                        gym.ObservationWrapper,
                        gym.ActionWrapper,
                        ActionObservationWrapper,
                    ),
                ), f"{type(wrapper)} is not an action/observation wrapper"
                wrappers.append(wrapper)
                wrapper = wrapper.env

        # Change the observation space
        self._action_space = spaces.Dict(
            {
                key: self.wrappers[key][0].action_space
                if len(self.wrappers[key]) > 0
                else self.mono_envs[key].action_space
                for key in self.keys
            }
        )
        self._observation_space = spaces.Dict(
            {
                key: self.wrappers[key][0].observation_space
                if len(self.wrappers[key]) > 0
                else self.mono_envs[key].observation_space
                for key in self.keys
            }
        )

    def action(self, actions: WrapperActType) -> ActType:
        new_action = {}
        for key in self.keys:
            action = actions[key]
            for wrapper in self.wrappers[key]:
                if isinstance(wrapper, (gym.ActionWrapper, ActionObservationWrapper)):
                    action = wrapper.action(action)
            new_action[key] = action

        return new_action

    def observation(self, observations: ObsType) -> WrapperObsType:
        new_observation = {}
        for key in self.keys:
            observation = observations[key]
            for wrapper in reversed(self.wrappers[key]):
                if isinstance(
                    wrapper, (gym.ObservationWrapper, ActionObservationWrapper)
                ):
                    observation = wrapper.observation(observation)
            new_observation[key] = observation

        return new_observation
