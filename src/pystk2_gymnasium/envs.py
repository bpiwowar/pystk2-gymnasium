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
    return max([v.value for v in EnumType.Type.__members__.values()]) + 1


class STKRaceEnv(gym.Env[Any, STKAction]):
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

        STKRaceEnv.TRACKS = pystk2.list_tracks(pystk2.RaceConfig.RaceMode.NORMAL_RACE)

    def __init__(
        self,
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
        :param position: The position of the controlled kart, defaults to None
            for random, 0 to num_kart-1 assigns a rank, all the other values
            discard the controlled kart.
        :param max_paths: maximum number of paths ahead
        :param difficulty: difficulty (0 to 2)
        :param laps: Number of laps (default 1)
        """
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        STKRaceEnv.initialize(render_mode == "human")

        # Setup the variables
        self.default_track = track
        self.difficulty = difficulty
        self.rank_start = rank_start
        self.use_ai = use_ai
        self.laps = laps
        self.max_paths = max_paths
        self.num_kart = num_kart

        # Those will be set when the race is setup
        self.race = None
        self.world = None
        self.kart_ix = None
        self.current_track = None

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
                # Last attachment... is no attachment
                "attachment": spaces.Discrete(max_enum_value(pystk2.Attachment)),
                "skeed_factor": spaces.Box(
                    0.0, float("inf"), dtype=np.float32, shape=(1,)
                ),
                "attachment_time_left": spaces.Box(
                    0.0, float("inf"), dtype=np.float32, shape=(1,)
                ),
                "shield_time": spaces.Box(
                    0.0, float("inf"), dtype=np.float32, shape=(1,)
                ),
                "velocity": spaces.Box(
                    float("-inf"), float("inf"), dtype=np.float32, shape=(3,)
                ),
                "distance_down_track": spaces.Box(0.0, float("inf")),
                "front": spaces.Box(
                    -float("inf"), float("inf"), dtype=np.float32, shape=(3,)
                ),
                "jumping": spaces.Discrete(2),
                "items_position": spaces.Sequence(
                    spaces.Box(
                        -float("inf"), float("inf"), dtype=np.float32, shape=(3,)
                    )
                ),
                "items_type": spaces.Discrete(max_enum_value(pystk2.Item)),
                "karts_position": spaces.Sequence(
                    spaces.Box(
                        -float("inf"), float("inf"), dtype=np.float32, shape=(3,)
                    )
                ),
                "paths_distance": spaces.Sequence(
                    spaces.Box(0, float("inf"), dtype=np.float32, shape=(2,))
                ),
                "paths_width": spaces.Sequence(
                    spaces.Box(0, float("inf"), dtype=np.float32, shape=(1,))
                ),
                "paths_start": spaces.Sequence(
                    spaces.Box(0, float("inf"), dtype=np.float32, shape=(3,))
                ),
                "paths_end": spaces.Sequence(
                    spaces.Box(0, float("inf"), dtype=np.float32, shape=(3,))
                ),
            }
        )

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

        return self.observation(), {}

    def step(
        self, action: STKAction
    ) -> tuple[pystk2.WorldState, float, bool, bool, dict[str, Any]]:
        if self.use_ai:
            self.race.step()
        else:
            self.race.step(
                pystk2.Action(
                    brake=action["brake"] > 0,
                    nitro=action["nitro"] > 0,
                    drift=action["drift"] > 0,
                    rescue=action["rescue"] > 0,
                    fire=action["fire"] > 0,
                )
            )

        kart = self.world.karts[self.kart_ix]
        distance = kart.overall_distance
        terminated = kart.finish_time > 0
        obs = self.observation()
        reward = kart.overall_distance - distance

        # --- Find the track
        return obs, reward, terminated, False, {}

    def render(self):
        # Just do nothing... rendering is done directly
        pass

    def observation(self):
        self.world.update()
        kart = self.world.karts[self.kart_ix]

        def kartview(x):
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

        # Sort closest to front
        x_front = kartview(kart.front)

        def sort_closest(positions, *lists):
            distances = [(p - x_front) @ x_front for p in positions]

            # Change distances: if d < 0, d <- -d+max_d+1
            # so that negative distances are after
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
            if ix != self.kart_ix
        ]
        sort_closest(karts_position)

        items_position = [kartview(item.location) for item in self.world.items]
        items_type = [item.type.value for item in self.world.items]
        sort_closest(items_position, items_type)

        return {
            # Kart properties
            "powerup": kart.powerup.num,
            "attachment": kart.attachment.type.value,
            "attachment_time_left": np.array(
                [kart.attachment.time_left], dtype=np.float32
            ),
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
            "items_position": items_position,
            "items_type": items_type,
            # Other karts (kart point of view)
            "karts_position": karts_position,
            # Paths
            "paths_distance": list(iterate_from(self.track.path_distance, path_ix)),
            "paths_width": list(iterate_from(self.track.path_width, path_ix)),
            "paths_start": list(
                x[0] for x in iterate_from(self.track.path_nodes, path_ix)
            ),
            "paths_end": list(
                x[1] for x in iterate_from(self.track.path_nodes, path_ix)
            ),
        }


class SimpleSTKRaceEnv(STKRaceEnv):
    def __init__(self, *, state_items=5, state_karts=5, state_paths=5, **kwargs):
        """A simpler race environment with fixed width data

        :param state_items: The number of items, defaults to 5
        :param state_karts: The number of karts, defaults to 5
        """
        super().__init__(max_paths=state_paths, **kwargs)
        self.state_items = state_items
        self.state_karts = state_karts
        self.state_paths = state_paths

        # Override some keys in the observation space
        space = self.observation_space

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
        space["items_type"] = spaces.Box(
            0, max_enum_value(pystk2.Item), dtype=np.int64, shape=(self.state_items,)
        )
        space["karts_position"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_karts, 3)
        )

    def make_tensor(self, state, name: str):
        value = state[name]
        space = self.observation_space[name]

        value = np.stack(value)
        assert (
            space.shape[1:] == value.shape[1:]
        ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"

        delta = space.shape[0] - value.shape[0]
        if delta > 0:
            shape = [delta] + list(space.shape[1:])
            value = np.concatenate([value, np.zeros(shape, dtype=space.dtype)], axis=0)
        elif delta < 0:
            value = value[:delta]

        assert (
            space.shape == value.shape
        ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"
        state[name] = value

    def observation(self):
        state = super().observation()

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


class DiscreteActionSTKRaceEnv(SimpleSTKRaceEnv):
    # Wraps the actions
    def __init__(self, acceleration_steps=10, steering_steps=10, **kwargs):
        super().__init__(**kwargs)
        self.acceleration_steps = acceleration_steps
        self.steering_steps = steering_steps

        self.action_space["acceleration"] = spaces.Discrete(acceleration_steps)
        self.action_space["steering"] = spaces.Discrete(steering_steps)

    def step(
        self, action: STKDiscreteAction
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action["acceleration"] = action["acceleration"] / self.acceleration_steps
        max_steer_angle = self.world.karts[self.kart_ix].max_steer_angle
        action["steering"] = action["steering"] / self.steering_steps * max_steer_angle
        return super().step(action)
