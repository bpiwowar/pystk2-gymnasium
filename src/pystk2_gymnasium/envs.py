import logging
import functools
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypedDict

import gymnasium as gym
import numpy as np
import pystk2
from gymnasium import spaces

from pystk2_gymnasium.utils import max_enum_value, rotate

logger = logging.getLogger("pystk2-gym")


float3D = Tuple[float, float, float]
float4D = Tuple[float, float, float, float]


@functools.lru_cache
def kart_action_space():
    return spaces.Dict(
        {
            # Acceleration
            "acceleration": spaces.Box(0, 1, shape=(1,)),
            # Steering angle
            "steer": spaces.Box(-1, 1, shape=(1,)),
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


@functools.lru_cache
def kart_observation_space():
    return spaces.Dict(
        {
            "powerup": spaces.Discrete(max_enum_value(pystk2.Powerup)),
            # Last attachment... is no attachment
            "attachment": spaces.Discrete(max_enum_value(pystk2.Attachment)),
            "skeed_factor": spaces.Box(0.0, float("inf"), dtype=np.float32, shape=(1,)),
            "energy": spaces.Box(0.0, float("inf"), dtype=np.float32, shape=(1,)),
            "attachment_time_left": spaces.Box(
                0.0, float("inf"), dtype=np.float32, shape=(1,)
            ),
            "shield_time": spaces.Box(0.0, float("inf"), dtype=np.float32, shape=(1,)),
            "velocity": spaces.Box(
                float("-inf"), float("inf"), dtype=np.float32, shape=(3,)
            ),
            "max_steer_angle": spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            "distance_down_track": spaces.Box(0.0, float("inf")),
            "front": spaces.Box(
                -float("inf"), float("inf"), dtype=np.float32, shape=(3,)
            ),
            "jumping": spaces.Discrete(2),
            "items_position": spaces.Sequence(
                spaces.Box(-float("inf"), float("inf"), dtype=np.float32, shape=(3,))
            ),
            "items_type": spaces.Sequence(spaces.Discrete(max_enum_value(pystk2.Item))),
            "karts_position": spaces.Sequence(
                spaces.Box(-float("inf"), float("inf"), dtype=np.float32, shape=(3,))
            ),
            "paths_distance": spaces.Sequence(
                spaces.Box(0, float("inf"), dtype=np.float32, shape=(2,))
            ),
            "paths_width": spaces.Sequence(
                spaces.Box(0, float("inf"), dtype=np.float32, shape=(1,))
            ),
            "paths_start": spaces.Sequence(
                spaces.Box(float("-inf"), float("inf"), dtype=np.float32, shape=(3,))
            ),
            "paths_end": spaces.Sequence(
                spaces.Box(float("-inf"), float("inf"), dtype=np.float32, shape=(3,))
            ),
        }
    )


class STKAction(TypedDict):
    # :> Acceleration, between 0 and 1
    acceleration: float
    # :> Steering, between -1 and 1 (but limited by max_steer)
    steering: float
    brake: bool
    drift: bool
    nitro: bool
    rescue: bool


def get_action(action: STKAction):
    return pystk2.Action(
        brake=int(action["brake"]) > 0,
        nitro=int(action["nitro"] > 0),
        drift=int(action["drift"] > 0),
        rescue=int(action["rescue"] > 0),
        fire=int(action["fire"] > 0),
        steer=float(action["steer"]),
        acceleration=float(action["acceleration"]),
    )


class BaseSTKRaceEnv(gym.Env[Any, STKAction]):
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
        if BaseSTKRaceEnv.INITIALIZED is None:
            BaseSTKRaceEnv.INITIALIZED = with_graphics
            pystk2.init(
                pystk2.GraphicsConfig.hd()
                if with_graphics
                else pystk2.GraphicsConfig.none()
            )

        assert (
            with_graphics == BaseSTKRaceEnv.INITIALIZED
        ), "Cannot switch from graphics to not graphics mode"

        BaseSTKRaceEnv.TRACKS = pystk2.list_tracks(
            pystk2.RaceConfig.RaceMode.NORMAL_RACE
        )

    def __init__(
        self,
        *,
        render_mode=None,
        track=None,
        num_kart=3,
        rank_start=None,
        use_ai=False,
        max_paths=None,
        laps: int = 1,
        difficulty: int = 2,
    ):
        """Creates a new race

        :param render_mode: Render mode, use "human" to watch the race, defaults
            to None
        :param track: Track to use (None = random)
        :param num_kart: Number of karts, defaults to 3
        :param max_paths: maximum number of paths ahead
        :param difficulty: AI bot skill level (from lowest 0 to highest 2)
        :param laps: Number of laps (default 1)
        """
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        BaseSTKRaceEnv.initialize(render_mode == "human")

        # Setup the variables
        self.default_track = track
        self.difficulty = difficulty
        self.laps = laps
        self.max_paths = max_paths
        self.num_kart = num_kart

        # Those will be set when the race is setup
        self.race = None
        self.world = None
        self.kart_ix = None
        self.current_track = None

    def reset_race(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pystk2.WorldState, Dict[str, Any]]:
        if self.race:
            del self.race

        # Set seed
        random = np.random.RandomState(seed)

        # Setup the race configuration
        self.current_track = self.default_track
        if self.current_track is None:
            self.current_track = self.TRACKS[random.randint(0, len(self.TRACKS))]
            logging.info("Selected %s", self.current_track)
        self.config = pystk2.RaceConfig(
            num_kart=self.num_kart,
            seed=seed or 0,
            difficulty=self.difficulty,
            track=self.current_track,
            laps=self.laps,
        )

        for ix in range(self.num_kart):
            if ix > 0:
                self.config.players.append(pystk2.PlayerConfig())
            self.config.players[
                ix
            ].controller = pystk2.PlayerConfig.Controller.AI_CONTROL

    def warmup_race(self):
        """Creates a new race and step until the first move"""
        assert self.race is None

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

    def close(self):
        super().close()
        if self.race is not None:
            self.race.stop()
            del self.race

    def get_observation(self, kart_ix):
        kart = self.world.karts[kart_ix]

        def kartview(x):
            """Returns a vector in the kart frame

            X right, Y up, Z forwards
            """
            return rotate(x - kart.location, kart.rotation)

        path_ix = next(
            ix[0]
            for ix, d in np.ndenumerate(self.track.path_distance[:, 1])
            if kart.distance_down_track <= d
        )

        def iterate_from(list, start):
            if self.max_paths is not None:
                size = min(len(list), self.max_paths)
            else:
                size = len(list)

            ix = start
            for _ in range(size):
                yield list[ix]
                ix += 1
                if ix >= size:
                    ix = 0

        def list_permute(list, sort_ix):
            list[:] = (list[ix] for ix in sort_ix)

        def sort_closest(positions, *lists):
            distances = [np.linalg.norm(p) * np.sign(p[2]) for p in positions]

            # Change distances: if d < 0, d <- -d+max_d+1
            # so that negative distances are after positives ones
            max_d = max(distances)
            distances_2 = [d if d >= 0 else -d + max_d + 1 for d in distances]

            sorted_ix = np.argsort(distances_2)

            list_permute(positions, sorted_ix)
            for list in lists:
                list_permute(list, sorted_ix)

        # Sort items and karts by decreasing
        karts_position = [
            kartview(other_kart.location)
            for ix, other_kart in enumerate(self.world.karts)
            if ix != kart_ix
        ]
        sort_closest(karts_position)

        items_position = [kartview(item.location) for item in self.world.items]
        items_type = [item.type.value for item in self.world.items]
        sort_closest(items_position, items_type)

        # Add action if using AI bot
        obs = {}
        if self.use_ai:
            action = self.race.get_kart_action(kart_ix)
            obs = {
                "action": {
                    "steer": action.steer,
                    "brake": action.brake,
                    "nitro": action.nitro,
                    "drift": action.drift,
                    "rescue": action.rescue,
                    "fire": action.fire,
                    "acceleration": action.acceleration,
                }
            }

        return {
            **obs,
            # Kart properties
            "powerup": kart.powerup.num,
            "attachment": kart.attachment.type.value,
            "attachment_time_left": np.array(
                [kart.attachment.time_left], dtype=np.float32
            ),
            "max_steer_angle": np.array([kart.max_steer_angle], dtype=np.float32),
            "energy": np.array([kart.energy], dtype=np.float32),
            "skeed_factor": np.array([kart.skeed_factor], dtype=np.float32),
            "shield_time": np.array([kart.shield_time], dtype=np.float32),
            "jumping": 1 if kart.jumping else 0,
            # Kart physics (from the kart point view)
            "distance_down_track": np.array(
                [kart.distance_down_track], dtype=np.float32
            ),
            "velocity": kart.velocity_lc,
            "front": kartview(kart.front),
            # Items (kart point of view)
            "items_position": tuple(items_position),
            "items_type": tuple(items_type),
            # Other karts (kart point of view)
            "karts_position": tuple(karts_position),
            # Paths
            "paths_distance": tuple(iterate_from(self.track.path_distance, path_ix)),
            "paths_width": tuple(iterate_from(self.track.path_width, path_ix)),
            "paths_start": tuple(
                kartview(x[0]) for x in iterate_from(self.track.path_nodes, path_ix)
            ),
            "paths_end": tuple(
                kartview(x[1]) for x in iterate_from(self.track.path_nodes, path_ix)
            ),
        }


class STKRaceEnv(BaseSTKRaceEnv):
    """Single player race environment"""

    def __init__(self, *, rank_start=None, use_ai=False, **kwargs):
        """Creates a new race

        :param use_ai: Use STK built         AI bot instead of the agent action
        :param rank_start: The position of the controlled kart, defaults to None
            for random, 0 to num_kart-1 assigns a rank, all the other values
            discard the controlled kart.
        :param kwargs: General parameters, see BaseSTKRaceEnv
        """
        super().__init__(**kwargs)

        # Setup the variables
        self.rank_start = rank_start
        self.use_ai = use_ai

        # Those will be set when the race is setup
        self.kart_ix = None

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = kart_action_space()
        self.observation_space = kart_observation_space()

        if self.use_ai:
            self.observation_space["action"] = self.action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pystk2.WorldState, Dict[str, Any]]:

        super().reset_race()

        # Set the controlled kart position (if any)
        self.kart_ix = self.rank_start
        if self.kart_ix is None:
            self.kart_ix = np.random.randint(0, self.num_kart)
        logging.debug("Observed kart index %d", self.kart_ix)

        if self.use_ai:
            self.config.players[
                self.kart_ix
            ].camera_mode = pystk2.PlayerConfig.CameraMode.ON
        else:
            self.config.players[
                self.kart_ix
            ].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL

        self.warmup_race()
        self.world.update()

        return self.get_observation(self.kart_ix), {}

    def step(
        self, action: STKAction
    ) -> Tuple[pystk2.WorldState, float, bool, bool, Dict[str, Any]]:
        if self.use_ai:
            self.race.step()
        else:
            self.race.step(get_action(action))

        self.world.update()

        kart = self.world.karts[self.kart_ix]
        dt_m1 = max(kart.overall_distance, 0)
        terminated = kart.has_finished_race

        # Get the observation and update the world state
        obs = self.get_observation(self.kart_ix)

        d_t = max(0, kart.overall_distance)
        f_t = 1 if terminated else 0
        reward = (
            (d_t - dt_m1) / 10.0
            + (1.0 - kart.position / self.num_kart) * (3 + 7 * f_t)
            - 0.1
            + 10 * f_t
        )

        # --- Find the track
        return (
            obs,
            reward,
            terminated,
            False,
            {
                "position": kart.position,
                "distance": d_t,
            },
        )

    def render(self):
        # Just do nothing... rendering is done directly
        pass
