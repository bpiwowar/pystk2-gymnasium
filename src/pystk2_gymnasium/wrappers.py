"""
This module contains generic wrappers
"""

from copy import copy
from typing import Any, Callable, Dict, List, SupportsFloat, Tuple

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

from pystk2_gymnasium.definitions import ActionObservationWrapper, AgentException


class SpaceFlattener:
    """Flattens an observation or action space

    If the space has discrete and continuous values, returns
    a dictionary with "continuous" and "discrete" keys – each associated with
    a flattened observation or action. Otherwise, returns the flattened space itself.
    """

    def __init__(self, space: gym.Space):
        # Flatten the observation space
        self.continuous_keys = []
        self.shapes = []
        self.discrete_keys: List[str] = []
        self.indices = [0]

        continuous_size = 0
        lows = []
        highs = []
        counts = []

        # Combine keys (sort them beforehand so we always have the same order)
        for key, value in sorted(space.items(), key=lambda x: x[0]):
            # Ignore the AI action
            if key == "action":
                continue

            if isinstance(value, spaces.Discrete):
                self.discrete_keys.append(key)
                counts.append(value.n)
            elif isinstance(value, spaces.MultiDiscrete):
                self.discrete_keys.append(key)
                for n in value.nvec:
                    counts.append(n)
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

    def discrete(self, observation):
        """Concatenates discrete and multi-discrete keys"""
        r = []
        for key in self.discrete_keys:
            value = observation[key]
            if isinstance(value, int):
                r.append(value)
            else:
                r.extend(value)
        return r


class FlattenerWrapper(ActionObservationWrapper):
    """Flattens actions and observations."""

    def __init__(self, env: gym.Env, flatten_observations=True):
        super().__init__(env)

        self.flatten_observations = flatten_observations
        self.has_action = env.observation_space.get("action", None) is not None

        self.action_flattener = SpaceFlattener(env.action_space)
        self.action_space = self.action_flattener.space

        if flatten_observations:
            self.observation_flattener = SpaceFlattener(env.observation_space)
            self.observation_space = self.observation_flattener.space
        elif self.has_action:
            self.observation_space = copy(env.observation_space)
            self.observation_space["action"] = self.action_flattener.space

    def observation(self, observation):
        if self.flatten_observations:
            new_obs = {
                "discrete": np.array(self.observation_flattener.discrete(observation)),
                "continuous": np.concatenate(
                    [
                        observation[key].flatten()
                        for key in self.observation_flattener.continuous_keys
                    ]
                ),
            }
        elif self.has_action:
            new_obs = {key: value for key, value in observation.items()}
        else:
            return observation

        if self.has_action:
            # Transforms from nested action to a flattened
            obs_action = observation["action"]
            discrete = np.array(self.action_flattener.discrete(obs_action))
            if self.action_flattener.only_discrete:
                new_obs["action"] = discrete
            else:
                continuous = np.concatenate(
                    [
                        (
                            np.array([obs_action[key]])
                            if isinstance(obs_action[key], float)
                            else obs_action[key].flatten()
                        )
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
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        raise RuntimeError("Should not be called - fake mono environment")


class MonoAgentWrapperAdapter(ActionObservationWrapper):
    """Adapts a mono agent wrapper for a multi-agent one"""

    def __init__(
        self,
        env: gym.Env,
        *,
        keep_original=False,
        wrapper_factories: Dict[str, Callable[[gym.Env], Wrapper]],
    ):
        """Initialize an adapter that use distinct wrappers

        It supposes that the space/action space is a dictionary where each key
        corresponds to a different agent.

        :param env: The base environment
        :param keep_original: Keep original space
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
            try:
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
            except Exception as e:
                raise AgentException("Error when wrapping the environment", key) from e

        # Change the action/observation space
        observation_space = {
            key: (
                self.wrappers[key][0].observation_space
                if len(self.wrappers[key]) > 0
                else self.mono_envs[key].observation_space
            )
            for key in self.keys
        }
        action_space = {
            key: (
                self.wrappers[key][0].action_space
                if len(self.wrappers[key]) > 0
                else self.mono_envs[key].action_space
            )
            for key in self.keys
        }

        self.keep_original = keep_original
        if keep_original:
            for key, mono_env in self.mono_envs.items():
                observation_space[f"original/{key}"] = mono_env.observation_space

        # Set the action/observation space
        self._action_space = spaces.Dict(action_space)
        self._observation_space = spaces.Dict(observation_space)

    def action(self, actions: WrapperActType) -> ActType:
        new_action = {}
        for key in self.keys:
            try:
                action = actions[key]
                for wrapper in self.wrappers[key]:
                    if isinstance(
                        wrapper, (gym.ActionWrapper, ActionObservationWrapper)
                    ):
                        action = wrapper.action(action)
                new_action[key] = action
            except Exception as exc:
                raise AgentException(str(exc), key) from exc

        return new_action

    def observation(self, observations: ObsType) -> WrapperObsType:
        new_observation = {}
        for key in self.keys:
            try:
                observation = observations[key]
                if self.keep_original:
                    new_observation[f"original/{key}"] = observation

                for wrapper in reversed(self.wrappers[key]):
                    if isinstance(
                        wrapper, (gym.ObservationWrapper, ActionObservationWrapper)
                    ):
                        observation = wrapper.observation(observation)
                new_observation[key] = observation
            except Exception as exc:
                raise AgentException(str(exc), key) from exc

        return new_observation
