from typing import Any, ClassVar, List, Optional, TypedDict, Tuple
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
import pystk2


class STKAction(TypedDict):
    #:> Acceleration, between 0 and 1
    acceleration: float
    #:> Acceleration, between 0 and 1
    steering: float
    brake: bool
    drift: bool
    nitro: bool
    rescue: bool

float3D = Tuple[float, float, float]
float4D = Tuple[float, float, float, float]



class STKRaceEnv(gym.Env[pystk2.WorldState, STKAction]):
    metadata = {"render_modes": ["human"]}
    
    INITIALIZED: ClassVar[Optional[bool]] = None
    
    @staticmethod
    def initialize(with_graphics: bool):
        if STKRaceEnv.INITIALIZED is None:
            STKRaceEnv.INITIALIZED = with_graphics
            pystk2.init(pystk2.GraphicsConfig.hd() if with_graphics else pystk2.GraphicsConfig.none())

        assert with_graphics == STKRaceEnv.INITIALIZED, "Cannot switch from graphics to not graphics mode"


    def __init__(self, render_mode=None, num_kart=3, use_ai=False):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        STKRaceEnv.initialize(render_mode=="human")

        self.use_ai = use_ai

        # Initialize the pystk race
        config = pystk2.RaceConfig(num_kart=num_kart)
        config.players[0].controller = pystk2.PlayerConfig.Controller.AI_CONTROL if use_ai else pystk2.PlayerConfig.Controller.PLAYER_CONTROL

        for ix in range(1, num_kart):
            config.players.append(pystk2.PlayerConfig())
            config.players[ix].controller = pystk2.PlayerConfig.Controller.AI_CONTROL

        # ... and let's go!
        self.race = pystk2.Race(config)
        self.race.start()

        self.world = pystk2.WorldState()
        self.reset()

        # Observations are dictionaries with the agent's and the target's location.
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = Dict({
            # Acceleration
            "acceleration": Box(0, 1, shape=(1,)),
            # Steering angle
            "steering": Box(-1, 1, shape=(1,)),
            # Brake
            "brake": Discrete(2),
            # Drift
            "drift": Discrete(2),
            # Fire
            "fire": Discrete(2),
            # Nitro
            "nitro": Discrete(2),
            # Call the rescue bird
            "rescue": Discrete(2),
        })


    @staticmethod
    def convert_action(action: STKAction):
        pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[pystk2.WorldState, dict[str, Any]]:
        self.race.restart()
        while True:
            self.race.step()
            self.world.update()
            if self.world.phase == pystk2.WorldState.Phase.GO_PHASE:
                break


    def step(
        self, action: STKAction
    ) -> tuple[pystk2.WorldState, float, bool, bool, dict[str, Any]]:
        pystk_action = pystk2.Action(**action) if not self.use_ai else None
        self.race.step(pystk_action)

        # Get the new state
        self.world.update()

        # And output it
        obs = {}
        reward = 0
        terminated = self.world.karts[0].finish_time > 0

        return self.world, reward, terminated, False, {}
    
    def render(self):
        # Just do nothing... rendering is done directly
        pass


class STKKartObs(TypedDict):
    location: float3D
    front: float3D
    rotation: float4D

class STKObservation(TypedDict):
    powerup: int
    karts: List[STKKartObs]

class STKSimpleObserver(gym.ObservationWrapper[pystk2.WorldState, STKAction, STKObservation]):
    """Observation from a kart point of view"""
    def __init__(self, env, kart_ix=0):
        super().__init__(env)
        self.kart_ix = kart_ix

        self.observation_space = Dict({
            "powerup": Discrete(10),
            "attachment": Discrete(6),
            "attachment_time_left":  Box(0., float('inf'),  shape=(1,)),
            "shield_time": Box(0., float('inf'),  shape=(1,)),
            "velocity": Box(0., float('inf'), shape=(3,)),
            # "items": [],
            # "karts": []
        })

    """Simplify the observations"""
    def observation(self, state: pystk2.WorldState) -> STKObservation:
        kart = state.karts[self.kart_ix]

        kart.powerup

        return state 
        # {
        #     "powerup": kart.powerup.num,
        #     "attachment": kart.attachment.type.value,
        #     "attachment_time_left": kart.attachment.time_left,
        #     "shield_time": kart.shield_time,
        # }

if __name__ == "__main__":
    env = STKSimpleObserver(STKRaceEnv(render_mode="human"))
    for _ in range(100):
        state, reward, terminated, done, *_ = env.step({"acceleration": .9})