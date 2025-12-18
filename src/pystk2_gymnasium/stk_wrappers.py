"""
This module contains STK-specific wrappers
"""

import copy
import logging
import sys
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pystk2
from gymnasium import spaces

from .envs import STKAction, STKRaceEnv
from .definitions import ActionObservationWrapper
from pystk2_gymnasium.utils import Discretizer, max_enum_value


class PolarObservations(gym.ObservationWrapper):
    """Modifies position to polar positions

    Angles are in radian

    input: X right, Y up, Z forwards
    output: (angle in the ZX plane, angle in the ZY plane, distance)
    """

    #: Keys to transform (batch)
    KEYS = ["items_position", "karts_position", "paths_start", "paths_end"]
    #: Keys to transform (single)
    SIMPLE_KEYS = ["center_path"]

    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, **kwargs)

    def observation(self, obs):
        # Shallow copy
        obs = {**obs}

        for key in PolarObservations.SIMPLE_KEYS:
            v = obs[key]
            distance = np.linalg.norm(v, axis=None)
            angle_zx = np.arctan2(v[0], v[2])
            angle_zy = np.arctan2(v[1], v[2])
            v[:] = angle_zx, angle_zy, distance

        for key in PolarObservations.KEYS:
            v = obs[key]

            is_tuple = False
            if isinstance(v, tuple):
                is_tuple = True
                v = np.stack(v)
            distance = np.linalg.norm(v, axis=1)
            angle_zx = np.arctan2(v[:, 0], v[:, 2])
            angle_zy = np.arctan2(v[:, 1], v[:, 2])
            v[:, 0], v[:, 1], v[:, 2] = angle_zx, angle_zy, distance

            if is_tuple:
                obs[key] = tuple(x for x in v)
        return obs


class ConstantSizedObservations(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        state_items=5,
        state_karts=5,
        state_paths=5,
        add_mask=False,
        **kwargs,
    ):
        """A simpler race environment with fixed width data

        :param state_items: The number of items, defaults to 5
        :param state_karts: The number of karts, defaults to 5
        """
        super().__init__(env, **kwargs)
        if isinstance(env.unwrapped, STKRaceEnv) and env.unwrapped.max_paths is None:
            logging.info("Setting unwrapped environment max_paths to %d", state_paths)
            env.unwrapped.max_paths = min(
                state_paths, env.unwrapped.max_paths or sys.maxsize
            )

        self.state_items = state_items
        self.state_karts = state_karts
        self.state_paths = state_paths

        # Override some keys in the observation space
        self._observation_space = space = copy.deepcopy(self.env.observation_space)

        space["paths_distance"] = spaces.Box(
            0, float("inf"), shape=(self.state_paths, 2), dtype=np.float32
        )
        space["paths_width"] = spaces.Box(
            0, float("inf"), shape=(self.state_paths, 1), dtype=np.float32
        )
        space["paths_start"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_paths, 3), dtype=np.float32
        )
        space["paths_end"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_paths, 3), dtype=np.float32
        )
        space["items_position"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_items, 3), dtype=np.float32
        )
        n_item_types = max_enum_value(pystk2.Item)
        space["items_type"] = spaces.MultiDiscrete(
            [n_item_types for _ in range(self.state_items)]
        )
        space["karts_position"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_karts, 3)
        )

        self.add_mask = add_mask
        if add_mask:
            space["paths_mask"] = spaces.Box(
                0, 1, shape=(self.state_paths,), dtype=np.int8
            )
            space["items_mask"] = spaces.Box(
                0, 1, shape=(self.state_items,), dtype=np.int8
            )
            space["karts_mask"] = spaces.Box(
                0, 1, shape=(self.state_karts,), dtype=np.int8
            )

    def make_tensor(self, state, name: str, default_value=0):
        value = state[name]
        space = self.observation_space[name]

        value = np.stack(value)
        assert (
            space.shape[1:] == value.shape[1:]
        ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"

        delta = space.shape[0] - value.shape[0]
        if delta > 0:
            shape = [delta] + list(space.shape[1:])
            value = np.concatenate(
                [value, np.full(shape, default_value, dtype=space.dtype)], axis=0
            )
        elif delta < 0:
            value = value[:delta]

        assert (
            space.shape == value.shape
        ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"
        state[name] = value

    def observation(self, state):
        # Shallow copy
        state = {**state}

        # Add masks
        def mask(length: int, size: int):
            v = np.zeros((size,), dtype=np.int8)
            v[:length] = 1
            return v

        if self.add_mask:
            state["paths_mask"] = mask(len(state["paths_width"]), self.state_paths)
            state["items_mask"] = mask(len(state["items_type"]), self.state_items)
            state["karts_mask"] = mask(len(state["karts_position"]), self.state_karts)

        # Ensures that the size of observations is constant
        self.make_tensor(state, "paths_distance")
        self.make_tensor(state, "paths_width")
        self.make_tensor(state, "paths_start")
        self.make_tensor(state, "paths_end")
        self.make_tensor(state, "items_position")
        self.make_tensor(state, "items_type")
        self.make_tensor(state, "karts_position")

        return state


class STKDiscreteAction(STKAction):
    acceleration: int
    steering: int


class DiscreteActionsWrapper(ActionObservationWrapper):
    # Wraps the actions
    def __init__(self, env: gym.Env, *, acceleration_steps=5, steer_steps=7, **kwargs):
        super().__init__(env, **kwargs)

        self._action_space = copy.deepcopy(env.action_space)

        self.d_acceleration = Discretizer(
            self.action_space["acceleration"], acceleration_steps
        )
        self._action_space["acceleration"] = self.d_acceleration.space

        self.d_steer = Discretizer(self.action_space["steer"], steer_steps)
        self._action_space["steer"] = self.d_steer.space

        if "action" in self.observation_space:
            # When using AI, "action" is part of the observation space
            self._observation_space = copy.deepcopy(self.observation_space)
            self._observation_space["action"]["steer"] = self.d_steer.space
            self._observation_space["action"][
                "acceleration"
            ] = self.d_acceleration.space

    def from_discrete(self, action):
        action = {**action}
        action["acceleration"] = self.d_acceleration.continuous(action["acceleration"])
        action["steer"] = self.d_steer.continuous(action["steer"])
        return action

    def to_discrete(self, action):
        action = {**action}
        action["acceleration"] = self.d_acceleration.discretize(action["acceleration"])
        action["steer"] = self.d_steer.discretize(action["steer"])
        return action

    def observation(self, obs):
        if "action" in obs:
            obs = {**obs}
            obs["action"] = self.to_discrete(obs["action"])
        return obs

    def action(
        self, action: STKDiscreteAction
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self.from_discrete(action)


class OnlyContinuousActionsWrapper(ActionObservationWrapper):
    """Removes the discrete actions"""

    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, **kwargs)

        self.discrete_actions = spaces.Dict(
            {
                key: value
                for key, value in env.action_space.items()
                if isinstance(value, spaces.Discrete)
            }
        )

        self._action_space = spaces.Dict(
            {
                key: value
                for key, value in env.action_space.items()
                if isinstance(value, spaces.Box)
            }
        )

    def observation(self, obs):
        if "action" in obs:
            obs = {**obs}
            obs["action"] = {
                key: obs["action"][key] for key in self.action_space.keys()
            }
        return obs

    def action(self, action: Dict) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return {**action, **{key: 0 for key, _ in self.discrete_actions.items()}}
