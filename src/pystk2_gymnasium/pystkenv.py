from typing import Any, ClassVar, List, Optional, Type, TypedDict, Tuple
import numpy as np
import logging

import gymnasium as gym
from gymnasium import spaces

import pystk2
from pystk2_gymnasium.utils import rotate

logger = logging.getLogger("pystk2-gym")


class STKAction(TypedDict):
    # :> Acceleration, between 0 and 1
    acceleration: float
    # :> Acceleration, between 0 and 1
    steering: float
    brake: bool
    drift: bool
    nitro: bool
    rescue: bool


float3D = Tuple[float, float, float]
float4D = Tuple[float, float, float, float]


def max_enum_value(EnumType: Type):
    """Returns the maximum enum value in a given enum type"""
    return max([v.value for v in EnumType.Type.__members__.values()])


class STKRaceEnv(gym.Env[pystk2.WorldState, STKAction]):
    metadata = {"render_modes": ["human"]}

    INITIALIZED: ClassVar[Optional[bool]] = None

    #: List of available tracks
    TRACKS: ClassVar[List[str]] = []

    #: Rank of the observed kart (random if None)
    rank_start: Optional[int]

    #: Use AI
    use_ai: bool

    @staticmethod
    def initialize(with_graphics: bool):
        if STKRaceEnv.INITIALIZED is None:
            STKRaceEnv.INITIALIZED = with_graphics
            pystk2.init(
                pystk2.GraphicsConfig.hd()
                if with_graphics
                else pystk2.GraphicsConfig.none()
            )

        assert (
            with_graphics == STKRaceEnv.INITIALIZED
        ), "Cannot switch from graphics to not graphics mode"

        STKRaceEnv.TRACKS = pystk2.list_tracks()

    def __init__(
        self,
        render_mode=None,
        track=None,
        num_kart=3,
        rank_start=None,
        use_ai=False,
        max_paths=None,
        difficulty: int = 2,
    ):
        """Creates a new race

        :param render_mode: Render mode, use "human" to watch the race, defaults
            to None
        :param track: Track to use (None = random)
        :param num_kart: Number of karts, defaults to 3
        :param position: The position of the controlled kart, defaults to None
            for random, 0 to num_kart-1 assigns a rank, all the other values
            discard the controlled kart.
        :param max_paths: maximum number of paths ahead
        :param difficulty: difficulty (0 to 2)
        """
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        STKRaceEnv.initialize(render_mode == "human")

        # Setup the variables
        self.kart_ix = None
        self.track = track
        self.difficulty = difficulty
        self.rank_start = rank_start
        self.use_ai = use_ai
        self.max_paths = max_paths
        self.num_kart = num_kart

        # ... and let's go!
        self.race = None
        self.world = None

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Dict(
            {
                # Acceleration
                "acceleration": spaces.Box(0, 1, shape=(1,)),
                # Steering angle
                "steering": spaces.Box(-1, 1, shape=(1,)),
                # Brake
                "brake": spaces.Discrete(2),
                # Drift
                "drift": spaces.Discrete(2),
                # Fire
                "fire": spaces.Discrete(2),
                # Nitro
                "nitro": spaces.Discrete(2),
                # Call the rescue bird
                "rescue": spaces.Discrete(2),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "powerup": spaces.Discrete(max_enum_value(pystk2.Powerup)),
                "attachment": spaces.Discrete(max_enum_value(pystk2.Attachment)),
                "attachment_time_left": spaces.Box(0.0, float("inf"), shape=(1,)),
                "shield_time": spaces.Box(0.0, float("inf"), shape=(1,)),
                "velocity": spaces.Box(0.0, float("inf"), shape=(3,)),
                # "items": spaces.Sequence(Box()),
                # "karts": []
            }
        )

    @staticmethod
    def convert_action(action: STKAction):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[pystk2.WorldState, dict[str, Any]]:
        if self.race:
            del self.race

        # Set seed
        random = np.random.RandomState(seed)

        # Setup the race configuration
        track = self.track
        if track is None:
            track = self.TRACKS[random.randint(0, len(self.TRACKS))]
        self.config = pystk2.RaceConfig(
            num_kart=self.num_kart, track=track, seed=seed, difficulty=self.difficulty
        )

        for ix in range(self.num_kart):
            if ix > 0:
                self.config.players.append(pystk2.PlayerConfig())
            self.config.players[
                ix
            ].controller = pystk2.PlayerConfig.Controller.AI_CONTROL

        # Set the controlled kart position (if any)
        self.kart_ix = self.rank_start
        if self.kart_ix is None:
            self.kart_ix = np.random.randint(0, self.num_kart)
        logging.debug("Observed kart index %d", self.kart_ix)

        if not self.use_ai:
            self.config.players[
                self.kart_ix
            ].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL

        self.race = pystk2.Race(self.config)

        # Start race
        self.race.start()
        self.world = pystk2.WorldState()
        self.track = pystk2.Track()
        self.track.update()

        while True:
            self.race.step()
            self.world.update()
            if self.world.phase == pystk2.WorldState.Phase.GO_PHASE:
                break

    def step(
        self, action: STKAction
    ) -> tuple[pystk2.WorldState, float, bool, bool, dict[str, Any]]:
        if self.use_ai:
            self.race.step()
        else:
            self.race.step(pystk2.Action(**action))

        # Get the new state
        self.world.update()

        # And output it
        reward = 0
        terminated = self.world.karts[0].finish_time > 0

        # --- Find the track
        return self.observation(), reward, terminated, False, {}

    def render(self):
        # Just do nothing... rendering is done directly
        pass

    def observation(self):
        kart = self.world.karts[self.kart_ix]

        def kartview(x):
            return rotate(x - kart.location, kart.rotation)

        path_ix = next(
            ix[0]
            for ix, d in np.ndenumerate(self.track.path_distance[:, 1])
            if kart.distance_down_track <= d
        )

        def iterate_from(list, start):
            size = len(list)
            ix = start
            for _ in range(size):
                yield list[ix]
                ix += 1
                if ix >= size:
                    ix = 0

        return {
            # Kart properties
            "powerup": kart.powerup.num,
            "attachment": kart.attachment.type.value,
            "attachment_time_left": kart.attachment.time_left,
            "shield_time": kart.shield_time,
            "jumping": 1 if kart.jumping else 0,
            # Kart physics (from the kart point view)
            "distance_down_track": kart.distance_down_track,
            "velocity": kart.velocity_lc,
            "front": kartview(kart.front),
            # Items (kart point of view)
            "items": [
                (kartview(item.location), item.type.value) for item in self.world.items
            ],
            # Other karts (kart point of view)
            "karts": [
                kartview(other_kart.location)
                for ix, other_kart in enumerate(self.world.karts)
                if ix != self.kart_ix
            ],
            # Tracks
            "tracks": [
                [
                    (distances, width, (kartview(x_start), kartview(x_end)))
                    for distances, width, (x_start, x_end) in zip(
                        iterate_from(self.track.path_distance, path_ix),
                        iterate_from(self.track.path_width, path_ix),
                        iterate_from(self.track.path_nodes, path_ix),
                    )
                ]
            ],
        }
