from enum import Enum
import heapq
import logging
import functools
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypedDict

import gymnasium as gym
import numpy as np
import pystk2
from gymnasium import spaces

from pystk2_gymnasium.pystk_process import (
    DirectSTKInterface,
    PySTKProcess,
    STKInterface,
)

from .utils import max_enum_value, rotate, rotate_batch
from .definitions import AgentSpec

logger = logging.getLogger("pystk2-gym")


# Global cache: track_name -> PathCache
_PATH_CACHE_REGISTRY: Dict[str, "PathCache"] = {}


def get_path_cache(track_name: str, track: "pystk2.Track") -> "PathCache":
    """Get or create a PathCache for a track (globally cached by track name)."""
    if track_name not in _PATH_CACHE_REGISTRY:
        _PATH_CACHE_REGISTRY[track_name] = PathCache(track)
    return _PATH_CACHE_REGISTRY[track_name]


class PathCache:
    """Cache for path traversal on a track.

    Memory-efficient iterator-based approach:
    - No per-node caching of full path lists
    - Uses heap for correct distance-based ordering
    - Returns an iterator that stops at max_paths or when cycling back

    Use get_path_cache() to get a globally cached instance by track name.
    """

    def __init__(self, track: "pystk2.Track"):
        """Initialize path cache for a track.

        :param track: The track object with path_distance and successors
        """
        self.num_nodes = len(track.path_distance)
        self.track = track
        self.successors = track.successors
        self.path_distance = track.path_distance
        self.track_length = track.length

        # Identify branch points (for has_branches property)
        self._branch_points: set = set()
        for ix in range(self.num_nodes):
            if len(self.successors[ix]) > 1:
                self._branch_points.add(ix)

    @property
    def has_branches(self) -> bool:
        """Return True if the track has branches."""
        return len(self._branch_points) > 0

    def iter_path_indices(self, start_ix: int, max_paths: Optional[int] = None):
        """Iterate over path indices starting from start_ix, ordered by distance.

        Uses heap-based traversal for correct distance ordering at branches.
        Yields nodes in order of distance from start. Stops when:
        - max_paths nodes have been yielded
        - All reachable nodes visited (cycle complete)

        :param start_ix: Starting node index
        :param max_paths: Maximum number of nodes to yield (None = all)
        """
        visited: set = set()
        count = 0

        path_distance = self.path_distance
        track_length = self.track_length
        start_dist = path_distance[start_ix, 1]

        def get_distance(ix: int) -> float:
            dist = path_distance[ix, 1]
            return max(abs(dist - start_dist), track_length / 2)

        # Use heap for distance-based ordering
        path_heap: List[Tuple[float, int]] = [(0.0, start_ix)]

        while path_heap and (max_paths is None or count < max_paths):
            _, current = heapq.heappop(path_heap)

            if current in visited:
                continue
            visited.add(current)
            yield current
            count += 1

            for succ in self.successors[current]:
                if succ not in visited:
                    heapq.heappush(path_heap, (get_distance(succ), succ))

    def get_path_indices(
        self, start_ix: int, max_paths: Optional[int] = None
    ) -> List[int]:
        """Get path indices as a list.

        :param start_ix: Starting node index
        :param max_paths: Maximum number of paths to return (None = all)
        :return: List of path indices in traversal order
        """
        return list(self.iter_path_indices(start_ix, max_paths))


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


class Phase(Enum):
    """A phase in PySTK (subset of STK phases)"""

    # 'Ready' is displayed
    READY_PHASE = 0

    # 'Set' is displayed
    SET_PHASE = 1

    # 'Go' is displayed, but this is already race phase
    GO_PHASE = 2

    # Other phases
    RACE_PHASE = 3

    @staticmethod
    def from_stk(source: pystk2.WorldState.Phase):
        if (source is None) or (source == pystk2.WorldState.Phase.READY_PHASE):
            return Phase.READY_PHASE
        if source == pystk2.WorldState.Phase.SET_PHASE:
            return Phase.SET_PHASE
        if source == pystk2.WorldState.Phase.GO_PHASE:
            return Phase.GO_PHASE
        return Phase.RACE_PHASE


@functools.lru_cache
def kart_observation_space(use_ai: bool):
    space = spaces.Dict(
        {
            "aux_ticks": spaces.Box(0.0, float("inf"), dtype=np.float32, shape=(1,)),
            "phase": spaces.Discrete(max_enum_value(Phase)),
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

    #: STK interface (either subprocess or direct)
    _stk: STKInterface = None

    def initialize(
        self, with_graphics: bool, use_subprocess: bool = True, graphics_config=None
    ):
        if self._stk is None:
            if use_subprocess:
                self._stk = PySTKProcess(with_graphics)
            else:
                self._stk = DirectSTKInterface(with_graphics, graphics_config)

        if not BaseSTKRaceEnv.TRACKS:
            BaseSTKRaceEnv.TRACKS = self._stk.list_tracks()

    def __del__(self):
        if self._stk:
            self._stk.close()

    def __init__(
        self,
        *,
        render_mode=None,
        track=None,
        num_kart=3,
        max_paths=None,
        laps: int = 1,
        difficulty: int = 2,
        use_subprocess: bool = True,
        num_cameras: int = 0,
        graphics_config=None,
        step_size: float = None,
    ):
        """Creates a new race

        :param render_mode: Render mode, use "human" to watch the race, defaults
            to None
        :param track: Track to use (None = random)
        :param num_kart: Number of karts, defaults to 3
        :param max_paths: maximum number of paths ahead
        :param difficulty: AI bot skill level (from lowest 0 to highest 2)
        :param laps: Number of laps (default 1)
        :param use_subprocess: If True, run STK in a subprocess (default).
            If False, run STK directly in the current process. Use False when
            running inside AsyncVectorEnv workers.
        :param num_cameras: Number of race cameras (default 0, max 8)
        :param graphics_config: Optional pystk2.GraphicsConfig to use instead
            of the default derived from render_mode
        :param step_size: Simulation time per physics tick in seconds.
            Defaults to the pystk2 default (0.1s).
        """
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.num_cameras = num_cameras
        self.initialize(render_mode == "human", use_subprocess, graphics_config)

        # Setup the variables
        self.default_track = track
        self.difficulty = difficulty
        self.laps = laps
        self.max_paths = max_paths
        self.num_kart = num_kart
        self.step_size = step_size

        # Those will be set when the race is setup
        self.race = None
        self.world = None
        self.current_track = None
        self.path_cache: Optional[PathCache] = None

    def reset_race(
        self,
        random: np.random.RandomState,
        *,
        options: Optional[Dict[str, Any]] = None,
    ):
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
        if self.step_size is not None:
            self.config.step_size = self.step_size
        if self.num_cameras > 0:
            self.config.num_cameras = self.num_cameras

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
        self.world = self._stk.get_world()
        return self.world

    def get_state(self, kart_ix: int, use_ai: bool):
        assert self.world is not None
        kart: pystk2.Kart = self.world.karts[kart_ix]
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
        assert self.world is not None

        kart: pystk2.Kart = self.world.karts[kart_ix]
        kart_location = np.array(kart.location, dtype=np.float32)
        kart_rotation = np.array(kart.rotation, dtype=np.float32)

        def kartview(x):
            """Returns a vector in the kart frame (single vector version)."""
            return rotate(
                np.asarray(x, dtype=np.float32) - kart_location, kart_rotation
            )

        def kartview_batch(positions: np.ndarray) -> np.ndarray:
            """Returns vectors in the kart frame (batch version)."""
            return rotate_batch(positions - kart_location, kart_rotation)

        # Use STK to get the kart path index
        path_ix = kart.node

        def sort_closest_batch(positions: np.ndarray, *lists):
            """Sort positions by distance, with positive z (front) first."""
            # z axis is front
            norms = np.linalg.norm(positions, axis=1)
            signs = np.sign(positions[:, 2])
            distances = norms * signs

            # Change distances: if d < 0, d <- -d+max_d+1
            # so that negative distances are after positive ones
            if len(distances) > 0:
                max_d = np.max(distances)
                distances = np.where(distances >= 0, distances, -distances + max_d + 1)
                sorted_ix = np.argsort(distances)
                positions = positions[sorted_ix]
                lists = tuple(lst[sorted_ix] for lst in lists)

            return positions, lists

        # --- Other karts (vectorized) ---
        other_kart_locs = np.array(
            [k.location for ix, k in enumerate(self.world.karts) if ix != kart_ix],
            dtype=np.float32,
        )
        if len(other_kart_locs) > 0:
            karts_position = kartview_batch(other_kart_locs)
            karts_position, _ = sort_closest_batch(karts_position)
        else:
            karts_position = np.empty((0, 3), dtype=np.float32)

        # --- Items (vectorized) ---
        if len(self.world.items) > 0:
            items_locs = np.array(
                [item.location for item in self.world.items], dtype=np.float32
            )
            items_type = np.array(
                [item.type.value for item in self.world.items], dtype=np.int32
            )
            items_position = kartview_batch(items_locs)
            items_position, (items_type,) = sort_closest_batch(
                items_position, items_type
            )
        else:
            items_position = np.empty((0, 3), dtype=np.float32)
            items_type = np.array([], dtype=np.int32)

        # --- Distance from center of track ---
        path_node = self.track.path_nodes[path_ix]
        path_endpoints = np.array([path_node[0], path_node[1]], dtype=np.float32)
        path_in_kart = kartview_batch(path_endpoints)
        start, end = path_in_kart[0], path_in_kart[1]

        s_e = start - end
        s_e_norm_sq = np.dot(s_e, s_e)
        if s_e_norm_sq > 0:
            x_orth = np.dot(s_e, start) * s_e / s_e_norm_sq - start
            center_path_distance = np.linalg.norm(x_orth) * np.sign(x_orth[0])
        else:
            x_orth = np.zeros(3, dtype=np.float32)
            center_path_distance = 0.0

        # Add action if using AI bot
        obs = {}
        if use_ai:
            action = self._stk.get_kart_action(kart_ix)
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

        # --- Path indices from cache ---
        path_indices = self.path_cache.get_path_indices(path_ix, self.max_paths)

        # --- Path positions (vectorized) ---
        num_paths = len(path_indices)
        if num_paths > 0:
            # Collect all path start/end points
            path_starts = np.array(
                [self.track.path_nodes[ix][0] for ix in path_indices], dtype=np.float32
            )
            path_ends = np.array(
                [self.track.path_nodes[ix][1] for ix in path_indices], dtype=np.float32
            )
            # Transform to kart view in batch
            paths_start = kartview_batch(path_starts)
            paths_end = kartview_batch(path_ends)
        else:
            paths_start = np.empty((0, 3), dtype=np.float32)
            paths_end = np.empty((0, 3), dtype=np.float32)

        return {
            **obs,
            # World properties
            "phase": Phase.from_stk(self.world.phase).value,
            "aux_ticks": np.array([self.world.aux_ticks], dtype=np.float32),
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
            "center_path": np.array(x_orth, dtype=np.float32),
            # Items (kart point of view)
            "items_position": tuple(items_position),
            "items_type": tuple(items_type),
            # Other karts (kart point of view)
            "karts_position": tuple(karts_position),
            # Paths
            "paths_distance": tuple(
                self.track.path_distance[ix] for ix in path_indices
            ),
            "paths_width": tuple(self.track.path_width[ix] for ix in path_indices),
            "paths_start": tuple(paths_start),
            "paths_end": tuple(paths_end),
        }

    def render(self):
        # Just do nothing... rendering is done directly
        pass

    def race_step(self, *action):
        return self._stk.race_step(*action)

    def warmup_race(self):
        self.track = self._stk.warmup_race(self.config)
        assert len(self.track.successors) == len(self.track.path_nodes)

        # Get path cache (globally cached by track name)
        self.path_cache = get_path_cache(self.current_track, self.track)

    def close(self):
        self._stk.close()


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
        if self.agent.kart:
            self.config.players[self.kart_ix].kart = self.agent.kart
        if self.agent.color > 0:
            self.config.players[self.kart_ix].color = self.agent.color

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

        :param agents: List of agent specifications.
        :param kwargs: General parameters, see BaseSTKRaceEnv
        """
        super().__init__(**kwargs)

        # Setup the variables
        self.agents = agents
        assert len(self.agents) <= self.num_kart, (
            f"Too many agents ({len(self.agents)}) for {self.num_kart} karts"
        )

        # Kart index for each agent (set when the race is setup)
        self.kart_indices = None

        ranked_agents = [agent for agent in agents if agent.rank_start is not None]
        used_ranks = set([agent.rank_start for agent in ranked_agents])

        assert all(agent.rank_start < self.num_kart for agent in ranked_agents), (
            "Karts must have all have a valid position"
        )
        assert len(set(ranked_agents)) == len(ranked_agents), (
            "Some agents have the same starting position"
        )

        self.free_positions = [
            ix for ix in range(self.num_kart) if ix not in used_ranks
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
            kart_ix = next(pos_iter) if agent.rank_start is None else agent.rank_start
            self.kart_indices.append(kart_ix)
            self.config.players[kart_ix].camera_mode = agent.camera_mode
            if not agent.use_ai:
                self.config.players[
                    kart_ix
                ].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL
            self.config.players[kart_ix].name = agent.name
            if agent.kart:
                self.config.players[kart_ix].kart = agent.kart
            if agent.color > 0:
                self.config.players[kart_ix].color = agent.color

        self.kart_m_indices = list(range(len(self.kart_indices)))
        self.kart_m_indices.sort(key=lambda ix: self.kart_indices[ix])
        logging.debug(
            "Observed kart indices %s / %s", self.kart_indices, self.kart_m_indices
        )

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
                for agent_ix, agent in zip(self.kart_m_indices, self.agents)
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
