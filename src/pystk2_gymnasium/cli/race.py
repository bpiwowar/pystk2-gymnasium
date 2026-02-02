"""Race command: load agents, run a race, output results."""

import importlib
import importlib.util
import json
import logging
import signal
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np

from pystk2_gymnasium.definitions import AgentException, AgentSpec

logger = logging.getLogger("pystk2.cli.race")


class ActionTimeoutError(Exception):
    """Raised when an agent takes too long to produce an action."""


@dataclass
class LoadedAgent:
    """An agent loaded from a source (zip, directory, or module)."""

    env_name: str
    player_name: str
    get_actor: Callable  # (state, obs_space, act_space) -> actor callable
    module_dir: Path  # directory containing pystk_actor.py
    get_wrappers: Optional[Callable]  # () -> list of extra wrappers
    source: str  # original source path/name
    load_error: Optional[str] = None  # set when the agent failed to load
    create_state: Callable = lambda: None  # () -> state (or None for stateless agents)


def _load_module_from_path(path: Path):
    """Load pystk_actor module from a directory path.

    If the directory is a Python package (contains __init__.py), uses
    importlib to support relative imports (e.g. ``from .actors import ...``).
    Otherwise, loads pystk_actor.py directly via spec_from_file_location.
    """
    actor_path = path / "pystk_actor.py"
    if not actor_path.exists():
        raise FileNotFoundError(f"No pystk_actor.py found in {path}")

    if (path / "__init__.py").exists():
        # Package with relative imports: use importlib with parent on sys.path
        parent = str(path.resolve().parent)
        pkg_name = path.name
        added = parent not in sys.path
        if added:
            sys.path.insert(0, parent)
        try:
            return importlib.import_module(f"{pkg_name}.pystk_actor")
        finally:
            if added:
                sys.path.remove(parent)

    spec = importlib.util.spec_from_file_location("pystk_actor", str(actor_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_agent(
    source: str,
    temp_dirs: List[tempfile.TemporaryDirectory],
    prepare_module_dir=None,
) -> LoadedAgent:
    """Load an agent from a zip file, directory, or Python module name.

    :param source: Path to zip/directory, or a Python module name.
        Supports ``@:CustomName`` suffix to override the player name.
    :param temp_dirs: List to append temp dirs to (kept alive for race duration)
    :param prepare_module_dir: Optional callback ``(path) -> None`` from the
        adapter, called on the module directory before importing.
    :returns: LoadedAgent
    """
    # Parse optional name override (e.g. "path/to/agent@:MyName")
    name_override = None
    if "@:" in source:
        source, name_override = source.rsplit("@:", 1)

    path = Path(source)

    if path.is_dir() and (path / "stk_actor" / "pystk_actor.py").is_file():
        # Repository root containing stk_actor/ package
        module_dir = path / "stk_actor"
        if prepare_module_dir:
            prepare_module_dir(module_dir)
        module = _load_module_from_path(module_dir)

    elif path.is_dir() and (path / "pystk_actor.py").exists():
        # Directory directly containing pystk_actor.py
        if prepare_module_dir:
            prepare_module_dir(path)
        module = _load_module_from_path(path)

    elif path.is_file() and path.suffix == ".zip":
        # Extract zip to temp dir
        tmp = tempfile.TemporaryDirectory(prefix="pystk2_agent_")
        temp_dirs.append(tmp)
        extract_dir = Path(tmp.name)
        with zipfile.ZipFile(str(path), "r") as zf:
            zf.extractall(str(extract_dir))
        logger.info("Extracted agent zip %s to %s", source, extract_dir)

        # Check if files are in a subdirectory
        entries = list(extract_dir.iterdir())
        if (
            len(entries) == 1
            and entries[0].is_dir()
            and (entries[0] / "pystk_actor.py").exists()
        ):
            extract_dir = entries[0]

        module = _load_module_from_path(extract_dir)

    elif not path.exists():
        raise FileNotFoundError(
            f"Agent source not found: {source} (not a valid path or module name)"
        )

    else:
        # Treat as Python module name (e.g. "stk_actor" on PYTHONPATH)
        try:
            full_module = f"{source}.pystk_actor"
            module = importlib.import_module(full_module)
        except ModuleNotFoundError as e:
            # Only fall back if the pystk_actor submodule itself is missing,
            # not if a transitive dependency (e.g. stable_baselines3) is missing
            if e.name != full_module:
                raise
            module = importlib.import_module(source)

    module_dir = Path(module.__file__).parent
    env_name = getattr(module, "env_name", "supertuxkart/full-v0")
    player_name = name_override or getattr(module, "player_name", source)
    get_wrappers = getattr(module, "get_wrappers", None)
    get_actor = module.get_actor
    create_state = getattr(module, "create_state", lambda: None)

    return LoadedAgent(
        env_name=env_name,
        player_name=player_name,
        get_actor=get_actor,
        module_dir=module_dir,
        get_wrappers=get_wrappers,
        source=source,
        create_state=create_state,
    )


def _build_wrapper_factory(loaded: LoadedAgent):
    """Build a wrapper factory for MonoAgentWrapperAdapter from an agent's env_name.

    Resolves the registered wrapper chain for the agent's env_name and returns
    a callable that applies those wrappers to a mono-agent environment.
    """
    from gymnasium.envs.registration import load_env_creator

    env_spec = gym.envs.registry.get(loaded.env_name)
    if env_spec is None:
        raise ValueError(f"Unknown environment: {loaded.env_name}")

    wrapper_specs = list(env_spec.additional_wrappers or [])

    # Collect extra wrappers from the agent (callables, not WrapperSpecs)
    extra_wrappers = []
    if loaded.get_wrappers is not None:
        extra_wrappers = loaded.get_wrappers() or []

    def factory(env):
        wrapped = env
        # Apply registered env wrappers (WrapperSpec objects)
        for ws in wrapper_specs:
            wrapper_cls = load_env_creator(ws.entry_point)
            wrapped = wrapper_cls(wrapped, **(ws.kwargs or {}))
        # Apply agent-provided wrappers (callables)
        for wrapper_fn in extra_wrappers:
            wrapped = wrapper_fn(wrapped)
        return wrapped

    return factory


def _call_with_timeout(func, args, timeout: Optional[float]):
    """Call func(*args) with an optional timeout (Unix only via SIGALRM).

    Falls back to a direct call (no timeout) when SIGALRM is unavailable
    or when called from a non-main thread (signals can only be set in the
    main thread).
    """
    if timeout is None:
        return func(*args)

    if not hasattr(signal, "SIGALRM"):
        # Non-Unix: just call without timeout
        return func(*args)

    import threading

    if threading.current_thread() is not threading.main_thread():
        # SIGALRM only works in the main thread
        return func(*args)

    def _handler(signum, frame):
        raise ActionTimeoutError(f"Agent action timed out after {timeout:.2f}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = func(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


def _load_adapter(path):
    """Load an adapter module from a Python file.

    The adapter must define ``create_actor(get_actor, module_dir, obs_space,
    act_space)`` which is responsible for loading state, calling get_actor,
    and returning a ready-to-call actor.

    It may also define ``prepare_module_dir(path)`` which is called before
    loading each agent module, allowing the adapter to prepare the directory
    (e.g. create missing ``__init__.py`` for legacy projects).
    """
    spec = importlib.util.spec_from_file_location("pystk2_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "create_actor"):
        raise AttributeError(f"Adapter file {path} must define a create_actor function")
    return module


def _apply_graphics_config(args, env_kwargs):
    """Apply --screen-width / --screen-height to env_kwargs if specified."""
    import pystk2

    screen_width = getattr(args, "screen_width", None)
    screen_height = getattr(args, "screen_height", None)
    if screen_width is not None or screen_height is not None:
        gfx = env_kwargs.get("graphics_config") or pystk2.GraphicsConfig.hd()
        if args.hide:
            gfx.display = False
        if screen_width is not None:
            gfx.screen_width = screen_width
        if screen_height is not None:
            gfx.screen_height = screen_height
        env_kwargs["graphics_config"] = gfx
        env_kwargs["use_subprocess"] = False


def _configure_recording(args, env_kwargs):
    """Configure env_kwargs for race recording.

    Uses 1280x720 by default unless --screen-width/--screen-height override.
    """
    import pystk2

    gfx = env_kwargs.get("graphics_config") or pystk2.GraphicsConfig.hd()
    # Default to 1280x720 for recordings (pystk2 presets use 600x400)
    if getattr(args, "screen_width", None) is None:
        gfx.screen_width = 1280
    if getattr(args, "screen_height", None) is None:
        gfx.screen_height = 720
    if args.hide:
        gfx.display = False
    env_kwargs["graphics_config"] = gfx
    env_kwargs["use_subprocess"] = False
    env_kwargs["num_cameras"] = min(args.cameras or args.num_karts, 8)
    render_sub_steps = getattr(args, "render_sub_steps", 1)
    if render_sub_steps > 1:
        default_step_size = 0.1  # pystk2 default
        env_kwargs["step_size"] = default_step_size / render_sub_steps


_CODEC_MAP = {
    ".mp4": "libx264",
    ".mkv": "libx264",
    ".avi": "mpeg4",
    ".webm": "libvpx",
    ".ogv": "libtheora",
    ".mov": "libx264",
}


class FrameRecorder:
    """Saves frames to a temporary directory to avoid holding them in memory.

    Frame durations are derived from game timestamps passed to
    :meth:`add_frame`.  Frames without a timestamp (e.g. the end card)
    use :attr:`END_CARD_DURATION`.
    """

    END_CARD_DURATION = 3.0  # seconds

    def __init__(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="pystk2_rec_")
        self._frame_count = 0
        self._frame_size = None
        self._timestamps: List[float] = []

    @property
    def frame_dir(self):
        return Path(self._tmpdir.name)

    def _save_frame(self, frame):
        """Save a single HxWx3 numpy array as a PNG file."""
        from PIL import Image

        path = self.frame_dir / f"{self._frame_count:06d}.png"
        Image.fromarray(frame).save(path)
        self._frame_count += 1

    def add_frame(self, frame, game_time: float = None):
        """Save a game frame.

        :param frame: HxWx3 numpy array.
        :param game_time: Game time in seconds from ``world.time``.
            When provided, the video will use per-frame durations derived
            from consecutive game timestamps instead of a fixed fps.
        """
        if self._frame_size is None:
            self._frame_size = (frame.shape[1], frame.shape[0])
        if game_time is not None:
            self._timestamps.append(game_time)
        self._save_frame(frame)

    def capture_sub_steps(self, env, sub_steps):
        """Run sub_steps-1 extra physics ticks and capture each frame.

        Must be called *after* ``env.step()`` has been called and the
        first frame has already been captured.  Retrieves the last applied
        actions via ``race.get_kart_action()`` and replays them.

        :param env: The gymnasium environment.
        :param sub_steps: Total sub-steps per action (including the
            ``env.step()`` tick).  When <= 1 this is a no-op.
        """
        if sub_steps <= 1:
            return
        unwrapped = env.unwrapped
        base_time = unwrapped.world.time
        step_size = unwrapped.config.step_size

        # Retrieve the last-applied actions for player-controlled karts,
        # sorted by kart index (the order race.step() expects).
        race = unwrapped._stk.race
        player_kart_indices = sorted(
            kart_ix
            for agent, kart_ix in zip(unwrapped.agents, unwrapped.kart_indices)
            if not agent.use_ai
        )
        pystk_actions = [race.get_kart_action(k) for k in player_kart_indices]

        for i in range(sub_steps - 1):
            race.step(pystk_actions)
            screen = race.screen_capture()
            if screen is not None and screen.size > 0:
                self.add_frame(
                    np.array(screen),
                    game_time=base_time + (i + 1) * step_size,
                )

    @staticmethod
    def _load_font(size):
        """Load a TrueType font at the given size, trying common paths."""
        from PIL import ImageFont

        candidates = [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            # macOS system fonts
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        # Pillow >= 10.0 load_default accepts a size argument
        return ImageFont.load_default(size=size)

    def add_end_card(self, track_name, results):
        """Add a single end-card frame showing track name and agent results.

        :param track_name: Name of the track.
        :param results: List of dicts with keys ``name``, ``start_pos``,
            ``end_pos`` for each controlled agent.
        """
        from PIL import Image, ImageDraw

        if self._frame_size is None:
            return

        width, height = self._frame_size
        img = Image.new("RGB", (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Scale fonts relative to frame height
        title_size = max(20, height // 10)
        body_size = max(12, height // 20)

        font_title = self._load_font(title_size)
        font_body = self._load_font(body_size)

        # Track name at the very top
        draw.text(
            (width / 2, title_size * 0.7),
            track_name or "Unknown Track",
            fill=(255, 255, 255),
            font=font_title,
            anchor="mm",
        )

        # Results in two columns: "#{start}. {name} (end #{end})"
        line_height = body_size + 6
        top_y = title_size * 1.5 + line_height
        col_x = [width * 0.05, width * 0.52]
        n_rows = (len(results) + 1) // 2  # rows needed for two columns

        for i, r in enumerate(results):
            col = i // n_rows  # fill left column first, then right
            row = i % n_rows
            start = r["start_pos"]
            end = r.get("end_pos")
            if end is not None:
                line = f"#{start}. {r['name']} (end #{end})"
            else:
                line = f"#{start}. {r['name']}"

            draw.text(
                (col_x[col], top_y + row * line_height),
                line,
                fill=(200, 200, 200),
                font=font_body,
                anchor="lm",
            )

        self._save_frame(np.array(img))

    def _compute_durations(self, n_frames):
        """Compute per-frame durations from game timestamps.

        Returns a list of durations (one per frame) or ``None`` if fewer
        than two timestamps were recorded.
        """
        ts = self._timestamps
        if len(ts) < 2:
            return None

        # Durations between consecutive game frames
        durations = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]

        # Clamp to avoid zero or negative durations from float imprecision
        min_dur = 1e-4
        durations = [max(d, min_dur) for d in durations]

        # The last game frame gets the same duration as the previous one
        durations.append(durations[-1])

        # Extra frames beyond timestamps (e.g. end card)
        while len(durations) < n_frames:
            durations.append(self.END_CARD_DURATION)

        return durations[:n_frames]

    def save(self, record_path):
        """Assemble saved frames into a video file."""
        if self._frame_count == 0:
            return
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        # Collect sorted frame paths as strings
        frame_paths = sorted(str(p) for p in self.frame_dir.glob("*.png"))
        path = Path(record_path)
        codec = _CODEC_MAP.get(path.suffix.lower(), "libx264")
        logger.info(
            "Generating video from %d frames (%s)...",
            self._frame_count,
            path,
        )

        durations = self._compute_durations(len(frame_paths))
        if durations is not None:
            clip = ImageSequenceClip(frame_paths, durations=durations)
        else:
            # Fallback: no timestamps, assume 10 fps (STK default step_size)
            clip = ImageSequenceClip(frame_paths, fps=10)

        # write_videofile needs fps even when durations are set;
        # derive from the median frame duration.
        if clip.fps is None:
            median_dur = sorted(durations)[len(durations) // 2]
            clip.fps = max(1, round(1.0 / median_dur))

        clip.write_videofile(str(path), codec=codec, logger="bar")
        logger.info("Saved recording to %s", path)

    def cleanup(self):
        self._tmpdir.cleanup()

    def __del__(self):
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass


def _assign_karts_and_colors(n):
    """Return lists of (kart, color) for *n* agents.

    Must be called after pystk2 is initialized (e.g. after gym.make).
    Cycles through available kart models and spreads colors evenly
    across the hue wheel so that every agent looks distinct.
    """
    import pystk2

    karts = pystk2.list_karts()
    if not karts:
        return [("", 0.0)] * n

    assigned = []
    for ix in range(n):
        kart = karts[ix % len(karts)]
        # Spread hue evenly; skip 0.0 which means "default color"
        color = ((ix + 1) / (n + 1)) if n > 1 else 0.0
        assigned.append((kart, color))
    return assigned


def _create_actors(loaded_agents, env, adapter_module):
    """Create actor callables from loaded agents, wrapping errors."""
    create_actor = (
        getattr(adapter_module, "create_actor", None) if adapter_module else None
    )
    actors = []
    for ix, la in enumerate(loaded_agents):
        key = str(ix)
        obs_space = env.observation_space[key]
        act_space = env.action_space[key]
        try:
            if create_actor is not None:
                actor = create_actor(la.get_actor, la.module_dir, obs_space, act_space)
            else:
                actor = la.get_actor(la.module_dir, obs_space, act_space)
        except Exception as exc:
            raise AgentException("Exception when initializing the actor", key) from exc
        actors.append(actor)
        logger.info("Created actor for agent %d (%s)", ix, la.player_name)
    return actors


def _output_message(message, args):
    """Output a JSON message (results or error) to file and/or logger."""
    message_json = json.dumps(message, indent=2)
    logger.info("Race output:\n%s", message_json)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(message_json)
        logger.info("Written to %s", output_path)


def run_race(args):
    """Run a race with the given CLI arguments."""
    temp_dirs = []
    player_names = []
    try:
        _run_race_inner(args, temp_dirs, player_names)
    except AgentException as e:
        cause = e if e.__cause__ is None else e.__cause__
        tb = traceback.extract_tb(cause.__traceback__)
        key = int(e.key) if e.key is not None else -1
        message = {
            "type": "error",
            "key": key,
            "name": player_names[key] if key < len(player_names) else "?",
            "when": str(e),
            "message": str(cause),
            "traceback": traceback.format_list(tb),
        }
        _output_message(message, args)
    finally:
        for td in temp_dirs:
            td.cleanup()


def _run_race_inner(args, temp_dirs: list, player_names: list):  # noqa: C901
    from pystk2_gymnasium.wrappers import MonoAgentWrapperAdapter

    # --- Load adapter if specified (before agents, for prepare_module_dir) ---
    adapter_module = _load_adapter(args.adapter) if args.adapter else None
    prepare_module_dir = getattr(adapter_module, "prepare_module_dir", None)

    # --- Load agents ---
    loaded_agents: List[LoadedAgent] = []
    for ix, source in enumerate(args.agents):
        logger.info("Loading agent from %s", source)
        try:
            agent = load_agent(source, temp_dirs, prepare_module_dir=prepare_module_dir)
        except Exception as exc:
            raise AgentException("Exception when loading module", str(ix)) from exc
        logger.info(
            "Loaded agent %r (env=%s) from %s",
            agent.player_name,
            agent.env_name,
            source,
        )
        loaded_agents.append(agent)
        player_names.append(agent.player_name)

    num_agents = len(loaded_agents)

    # --- Build agent specs (kart/color assigned after pystk2 init) ---
    agent_specs = [AgentSpec(name=la.player_name) for la in loaded_agents]

    # --- Create the base multi-agent environment ---
    render_mode = None if args.hide else "human"
    env_kwargs = dict(
        agents=agent_specs,
        num_kart=args.num_karts,
        render_mode=render_mode,
        laps=args.laps,
    )
    if args.max_paths is not None:
        env_kwargs["max_paths"] = args.max_paths
    if args.track is not None:
        env_kwargs["track"] = args.track

    # --- Graphics / recording setup ---
    _apply_graphics_config(args, env_kwargs)
    if args.record:
        _configure_recording(args, env_kwargs)

    env = gym.make("supertuxkart/multi-full-v0", **env_kwargs)

    # Assign distinct karts and colors now that pystk2 is initialized
    kart_colors = _assign_karts_and_colors(num_agents)
    for spec, (kart, color) in zip(env.unwrapped.agents, kart_colors):
        spec.kart = kart
        spec.color = color

    # --- Apply per-agent wrappers ---
    wrapper_factories = {}
    for ix, la in enumerate(loaded_agents):
        wrapper_factories[str(ix)] = _build_wrapper_factory(la)

    env = MonoAgentWrapperAdapter(env, wrapper_factories=wrapper_factories)

    # --- Create actors (deferred: needs obs/action spaces) ---
    actors = _create_actors(loaded_agents, env, adapter_module)

    # --- Optional web visualization ---
    web_server = None
    if args.web:
        try:
            from pystk2_gymnasium.cli.stk_graph import WebDashboard

            agent_names = [la.player_name for la in loaded_agents]
            web_server = WebDashboard(
                port=args.web_port,
                num_controlled=num_agents,
                agent_names=agent_names,
            )
            web_server.start()
            logger.info("Web dashboard at http://localhost:%d", args.web_port)
        except ImportError:
            logger.error(
                "Web dashboard requires dash and plotly. "
                "Install with: pip install pystk2-gymnasium[web]"
            )
            sys.exit(1)

    # --- Race loop ---
    from tqdm import tqdm

    catch_errors = args.error_handling == "catch"
    action_timeout = args.action_timeout
    max_steps = args.max_steps
    controller = web_server.controller if web_server is not None else None

    recorder = None
    render_sub_steps = getattr(args, "render_sub_steps", 1)
    if args.record:
        recorder = FrameRecorder()
    action_times = [[] for _ in range(num_agents)]

    obs, info = env.reset()

    # Record starting grid positions (1-based)
    start_positions = {}
    kart_indices = getattr(env.unwrapped, "kart_indices", None)
    if kart_indices is not None:
        for ix, kart_ix in enumerate(kart_indices):
            start_positions[str(ix)] = kart_ix + 1

    done = False
    total_rewards = {str(ix): 0.0 for ix in range(num_agents)}
    finished = set()
    step_count = 0
    start_time = time.time()

    # Initialize per-agent state (None for stateless agents)
    states = [la.create_state() for la in loaded_agents]

    if web_server is not None:
        web_server.update(env, obs, info, total_rewards, step_count)

    pbar = tqdm(
        total=max_steps,
        desc="Racing",
        unit="step",
    )

    try:
        while not done:
            # When the web UI is active, wait for step/run/stop
            if controller is not None and not controller.wait_for_step():
                logger.info("Race stopped from web UI")
                break

            actions = {}
            for ix, actor in enumerate(actors):
                key = str(ix)
                try:
                    t_start = time.perf_counter()
                    action = _call_with_timeout(
                        actor, (states[ix], obs[key]), action_timeout
                    )
                    action_times[ix].append(time.perf_counter() - t_start)
                    actions[key] = action
                except Exception as exc:
                    if not catch_errors:
                        raise AgentException(
                            f"Agent {ix} ({loaded_agents[ix].player_name}): {exc}",
                            key,
                        ) from exc
                    logger.warning(
                        "Agent %d (%s) error: %s — using random action",
                        ix,
                        loaded_agents[ix].player_name,
                        exc,
                    )
                    actions[key] = env.action_space[key].sample()

            obs, reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            done = terminated or truncated
            if max_steps is not None and step_count >= max_steps:
                done = True

            # Track which agents have finished
            agent_terminated = info.get("terminated", {})
            for key, t in agent_terminated.items():
                if t:
                    finished.add(key)
            pbar.update(1)
            pbar.set_postfix(finished=f"{len(finished)}/{num_agents}")

            # Capture frames for recording
            if recorder is not None:
                screen = env.unwrapped._stk.race.screen_capture()
                if screen is not None and screen.size > 0:
                    recorder.add_frame(
                        np.array(screen),
                        game_time=env.unwrapped.world.time,
                    )
                recorder.capture_sub_steps(env, render_sub_steps)

            # Accumulate per-agent rewards
            agent_rewards = info.get("reward", {})
            for key, r in agent_rewards.items():
                total_rewards[key] = total_rewards.get(key, 0.0) + float(r)

            if web_server is not None:
                web_server.update(env, obs, info, total_rewards, step_count)

    except KeyboardInterrupt:
        logger.info("Race interrupted by user")
    finally:
        pbar.close()
        elapsed = time.time() - start_time
        env.close()

        if recorder is not None:
            agent_infos = info.get("infos", {})
            track_name = getattr(env.unwrapped, "current_track", args.track)
            end_card_results = []
            for ix, la in enumerate(loaded_agents):
                key = str(ix)
                end_pos = agent_infos.get(key, {}).get("position")
                end_card_results.append(
                    {
                        "name": la.player_name,
                        "start_pos": start_positions.get(key, "?"),
                        "end_pos": end_pos,
                    }
                )
            recorder.add_end_card(track_name, end_card_results)
            recorder.save(args.record)
            recorder.cleanup()

    # --- Build results ---
    agent_infos = info.get("infos", {})
    rewards = info.get("reward", {})
    results_payload = []
    message = {
        "type": "results",
        "track": getattr(env.unwrapped, "current_track", args.track),
        "steps": step_count,
        "elapsed_seconds": round(elapsed, 2),
        "results": results_payload,
    }
    for ix, la in enumerate(loaded_agents):
        key = str(ix)
        agent_info = agent_infos.get(key, {})
        avg_action_time = float(np.mean(action_times[ix])) if action_times[ix] else 0.0
        results_payload.append(
            {
                "key": ix,
                "name": la.player_name,
                "reward": rewards.get(key, total_rewards.get(key, 0.0)),
                "position": agent_info.get("position", None),
                "avg_action_time": avg_action_time,
            }
        )
        logger.info(
            "Agent %d (%s): avg action time = %.4fs",
            ix,
            la.player_name,
            avg_action_time,
        )

    _output_message(message, args)
