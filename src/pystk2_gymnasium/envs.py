import logging
import functools
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypedDict

import gymnasium as gym
import numpy as np
import pystk2
from gymnasium import spaces

from pystk2_gymnasium.pystk_process import PySTKProcess

from .utils import max_enum_value, rotate
from .definitions import AgentSpec

logger = logging.getLogger("pystk2-gym")


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
def kart_observation_space(use_ai: bool):
    space = spaces.Dict(
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
            "distance_down_track": spaces.Box(-float("inf"), float("inf")),
            "center_path_distance": spaces.Box(
                float("-inf"), float("inf"), dtype=np.float32, shape=(1,)
            ),
            "center_path": spaces.Box(
                -float("inf"), float("inf"), dtype=np.float32, shape=(3,)
            ),
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

    if use_ai:
        space["action"] = kart_action_space()

    return space


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

    #: List of available tracks
    TRACKS: ClassVar[List[str]] = []

    #: Flag when pystk is initialized
    _process: PySTKProcess = None

    def initialize(self, with_graphics: bool):
        if self._process is None:
            self._process = PySTKProcess(with_graphics)

        if not BaseSTKRaceEnv.TRACKS:
            BaseSTKRaceEnv.TRACKS = self._process.list_tracks()

    def __init__(
        self,
        *,
        render_mode=None,
        track=None,
        num_kart=3,
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
        self.initialize(render_mode == "human")

        # Setup the variables
        self.default_track = track
        self.difficulty = difficulty
        self.laps = laps
        self.max_paths = max_paths
        self.num_kart = num_kart

        # Those will be set when the race is setup
        self.race = None
        self.world = None
        self.current_track = None

    def reset_race(
        self,
        random: np.random.RandomState,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pystk2.WorldState, Dict[str, Any]]:
        if self.race:
            self.race = None

        # Setup the race configuration
        self.current_track = self.default_track
        if self.current_track is None:
            self.current_track = self.TRACKS[random.randint(0, len(self.TRACKS))]
            logging.debug("Selected %s", self.current_track)
        self.config = pystk2.RaceConfig(
            num_kart=self.num_kart,
            seed=random.randint(2**16),
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

    def world_update(self, keep=True):
        """Update world state, but keep some information to compute reward"""
        if keep:
            self.last_overall_distances = [
                max(kart.overall_distance, 0) for kart in self.world.karts
            ]
        self.world = self._process.get_world()
        return self.world

    def get_state(self, kart_ix: int, use_ai: bool):
        kart = self.world.karts[kart_ix]
        terminated = kart.has_finished_race

        # Get the observation and update the world state
        obs = self.get_observation(kart_ix, use_ai)

        d_t = max(0, kart.overall_distance)
        f_t = 1 if terminated else 0
        reward = (
            (d_t - self.last_overall_distances[kart_ix]) / 10.0
            + (1.0 - kart.position / self.num_kart) * (3 + 7 * f_t)
            - 0.1
            + 10 * f_t
        )
        return (
            obs,
            reward,
            terminated,
            {
                "position": kart.position,
                "distance": d_t,
            },
        )

    def get_observation(self, kart_ix, use_ai):
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
            # z axis is front
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

        # Distance from center of track
        start, end = kartview(self.track.path_nodes[path_ix][0]), kartview(
            self.track.path_nodes[path_ix][1]
        )

        s_e = start - end
        x_orth = np.dot(s_e, start) * s_e / np.linalg.norm(s_e) ** 2 - start

        center_path_distance = np.linalg.norm(x_orth) * np.sign(x_orth[0])

        # Add action if using AI bot
        # (this corresponds to the action before the observation)
        obs = {}
        if use_ai:
            # Adds actions
            action = self._process.get_kart_action(kart_ix)
            obs = {
                "action": {
                    "acceleration": np.array([action.acceleration], dtype=np.float32),
                    "brake": action.brake,
                    "drift": action.drift,
                    "fire": action.fire,
                    "nitro": action.nitro,
                    "rescue": action.rescue,
                    "steer": np.array([action.steer], dtype=np.float32),
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
            # path center
            "center_path_distance": np.array([center_path_distance], dtype=np.float32),
            "center_path": np.array(x_orth),
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

    def render(self):
        # Just do nothing... rendering is done directly
        pass

    def race_step(self, *action):
        return self._process.race_step(*action)

    def warmup_race(self):
        self.track = self._process.warmup_race(self.config)

    def close(self):
        self._process.close()


class STKRaceEnv(BaseSTKRaceEnv):
    """Single player race environment"""

    #: Use AI
    spec: AgentSpec

    def __init__(self, *, agent: Optional[AgentSpec] = None, **kwargs):
        """Creates a new race

        :param spec: Agent spec
        :param kwargs: General parameters, see BaseSTKRaceEnv
        """
        super().__init__(**kwargs)

        # Setup the variables
        self.agent = agent if agent is not None else AgentSpec()

        # Those will be set when the race is setup
        self.kart_ix = None

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = kart_action_space()
        self.observation_space = kart_observation_space(self.agent.use_ai)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pystk2.WorldState, Dict[str, Any]]:
        random = np.random.RandomState(seed)

        super().reset_race(random, options=options)

        # Set the controlled kart position (if any)
        self.kart_ix = self.agent.rank_start
        if self.kart_ix is None:
            self.kart_ix = np.random.randint(0, self.num_kart)
        logging.debug("Observed kart index %d", self.kart_ix)

        # Camera setup
        self.config.players[
            self.kart_ix
        ].camera_mode = pystk2.PlayerConfig.CameraMode.ON
        self.config.players[self.kart_ix].name = self.agent.name

        if not self.agent.use_ai:
            self.config.players[
                self.kart_ix
            ].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL

        self.warmup_race()
        self.world_update(False)

        return self.get_observation(self.kart_ix, self.agent.use_ai), {}

    def step(
        self, action: STKAction
    ) -> Tuple[pystk2.WorldState, float, bool, bool, Dict[str, Any]]:
        if self.agent.use_ai:
            self.race_step()
        else:
            self.race_step(get_action(action))

        self.world_update()

        obs, reward, terminated, info = self.get_state(self.kart_ix, self.agent.use_ai)

        return (obs, reward, terminated, False, info)


class STKRaceMultiEnv(BaseSTKRaceEnv):
    """Multi-agent race environment"""

    def __init__(self, *, agents: List[AgentSpec] = None, **kwargs):
        """Creates a new race

        :param rank_start: The position of the controlled kart, defaults to None
            for random, 0 to num_kart-1 assigns a rank, all the other values
            discard the controlled kart.
        :param kwargs: General parameters, see BaseSTKRaceEnv
        """
        super().__init__(**kwargs)

        # Setup the variables
        self.agents = agents
        assert (
            len(self.agents) <= self.num_kart
        ), f"Too many agents ({len(self.agents)}) for {self.num_kart} karts"

        # Kart index for each agent (set when the race is setup)
        self.kart_indices = None

        ranked_agents = [agent for agent in agents if agent.rank_start is not None]

        assert all(
            agent.rank_start < self.num_kart for agent in ranked_agents
        ), "Karts must have all have a valid position"
        assert len(set(ranked_agents)) == len(
            ranked_agents
        ), "Some agents have the same starting position"

        self.free_positions = [
            ix for ix in range(self.num_kart) if ix not in ranked_agents
        ]

        self.action_space = spaces.Dict(
            {str(ix): kart_action_space() for ix in range(len(self.agents))}
        )
        self.observation_space = spaces.Dict(
            {
                str(ix): kart_observation_space(agent.use_ai)
                for ix, agent in enumerate(self.agents)
            }
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pystk2.WorldState, Dict[str, Any]]:

        random = np.random.RandomState(seed)

        super().reset_race(random, options=options)

        # Choose positions for agent karts
        np.random.shuffle(self.free_positions)
        pos_iter = iter(self.free_positions)
        self.kart_indices = []
        for agent in self.agents:
            kart_ix = agent.rank_start or next(pos_iter)
            self.kart_indices.append(kart_ix)
            self.config.players[kart_ix].camera_mode = agent.camera_mode
            if not agent.use_ai:
                self.config.players[
                    kart_ix
                ].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL
            self.config.players[kart_ix].name = agent.name

        logging.debug("Observed kart indices %s", self.kart_indices)

        self.warmup_race()
        self.world_update(False)

        return (
            {
                str(agent_ix): self.get_observation(kart_ix, agent.use_ai)
                for agent_ix, (agent, kart_ix) in enumerate(
                    zip(self.agents, self.kart_indices)
                )
            },
            {},
        )

    def step(
        self, actions: Dict[str, STKAction]
    ) -> Tuple[pystk2.WorldState, float, bool, bool, Dict[str, Any]]:
        # Performs the action
        assert len(actions) == len(self.agents)
        self.race_step(
            [
                get_action(actions[str(agent_ix)])
                for agent_ix, agent in enumerate(self.agents)
                if not agent.use_ai
            ]
        )

        # Update the world state
        self.world_update()

        observations = {}
        rewards = {}
        infos = {}
        multi_terminated = {}
        multi_done = {}
        terminated_count = 0
        for agent_ix, (agent, kart_ix) in enumerate(
            zip(self.agents, self.kart_indices)
        ):
            obs, reward, terminated, info = self.get_state(kart_ix, agent.use_ai)
            key = str(agent_ix)

            observations[key] = obs
            rewards[key] = reward
            multi_terminated[key] = terminated
            infos[key] = info

            if terminated:
                terminated_count += 1

        return (
            observations,
            # Only scalar rewards can be given: we sum them all
            np.sum(list(rewards.values())),
            terminated_count == len(self.agents),
            False,
            {
                "infos": infos,
                "done": multi_done,
                "terminated": multi_terminated,
                "reward": rewards,
            },
        )
