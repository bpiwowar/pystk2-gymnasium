"""
This module contains STK-specific wrappers
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import pystk2

import gymnasium as gym
from gymnasium.core import (
    Wrapper,
    WrapperActType,
    WrapperObsType,
    ObsType,
    ActType,
    SupportsFloat,
)

CameraMode = pystk2.PlayerConfig.CameraMode


class AgentException(Exception):
    """Exception for a given agent"""

    def __init__(self, message: str, key: str):
        super().__init__(message)
        self.key = key


@dataclass
class AgentSpec:
    #: The position of the controlled kart, defaults to None for random, 0 to
    # num_kart-1 assigns a rank, all the other values discard the controlled
    # kart.
    rank_start: Optional[int] = None
    #: Use the STK AI agent (ignores actions)
    use_ai: bool = False
    #: Player name
    name: str = ""
    #: Camera mode (AUTO, ON, OFF). By default, only non-AI agents get a camera
    camera_mode: CameraMode = CameraMode.AUTO
    #: Kart model name (empty string for default)
    kart: str = ""
    #: Kart color hue shift in [0, 1]. 0 uses the kart's default color.
    color: float = 0.0


class ActionObservationWrapper(Wrapper[ObsType, WrapperActType, ObsType, ActType]):
    """Combines action and observation wrapper"""

    def action(self, action: WrapperActType) -> ActType:
        raise NotImplementedError

    def observation(self, observation: ObsType) -> WrapperObsType:
        raise NotImplementedError

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the action wrapper."""
        Wrapper.__init__(self, env)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a
        modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: ActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using
        :meth:`self.observation` on the returned observations."""
        action = self.action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info
