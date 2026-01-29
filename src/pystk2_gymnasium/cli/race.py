"""Race command: load agents, run a race, output results."""

import importlib
import importlib.util
import json
import logging
import math
import signal
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

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
    state: Any  # loaded weights or None
    get_wrappers: Optional[Callable]  # () -> list of extra wrappers
    source: str  # original source path/name


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


def _load_weights(path: Path):
    """Try to load pystk_actor.pth from a directory; returns state or None."""
    weights_path = path / "pystk_actor.pth"
    if not weights_path.exists():
        return None
    try:
        import torch

        state = torch.load(str(weights_path), weights_only=True)
        logger.info("Loaded weights from %s", weights_path)
        return state
    except ImportError:
        logger.warning("torch not available, skipping weights file %s", weights_path)
        return None


def load_agent(
    source: str, temp_dirs: List[tempfile.TemporaryDirectory]
) -> LoadedAgent:
    """Load an agent from a zip file, directory, or Python module name.

    :param source: Path to zip/directory, or a Python module name.
        Supports ``@:CustomName`` suffix to override the player name.
    :param temp_dirs: List to append temp dirs to (kept alive for race duration)
    :returns: LoadedAgent
    """
    # Parse optional name override (e.g. "path/to/agent@:MyName")
    name_override = None
    if "@:" in source:
        source, name_override = source.rsplit("@:", 1)

    path = Path(source)

    if path.suffix == ".zip" and path.exists():
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
        state = _load_weights(extract_dir)

    elif path.is_dir() and (path / "pystk_actor.py").exists():
        module = _load_module_from_path(path)
        state = _load_weights(path)

    else:
        # Try as Python module name (e.g. "stk_actor" on PYTHONPATH)
        try:
            full_module = f"{source}.pystk_actor"
            module = importlib.import_module(full_module)
        except ModuleNotFoundError as e:
            # Only fall back if the pystk_actor submodule itself is missing,
            # not if a transitive dependency (e.g. stable_baselines3) is missing
            if e.name != full_module:
                raise
            module = importlib.import_module(source)
        # Load weights from the module's directory
        module_dir = Path(module.__file__).parent
        state = _load_weights(module_dir)

    env_name = getattr(module, "env_name", "supertuxkart/full-v0")
    player_name = name_override or getattr(module, "player_name", source)
    get_wrappers = getattr(module, "get_wrappers", None)
    get_actor = module.get_actor

    return LoadedAgent(
        env_name=env_name,
        player_name=player_name,
        get_actor=get_actor,
        state=state,
        get_wrappers=get_wrappers,
        source=source,
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
    """Call func(*args) with an optional timeout (Unix only via SIGALRM)."""
    if timeout is None:
        return func(*args)

    if not hasattr(signal, "SIGALRM"):
        # Non-Unix: just call without timeout
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


def _tile_frames(frames):
    """Tile a list of HxWx3 numpy arrays into a grid."""
    n = len(frames)
    if n == 1:
        return frames[0]
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    while len(frames) < nrows * ncols:
        frames.append(np.zeros_like(frames[0]))
    rows = []
    for r in range(nrows):
        rows.append(np.concatenate(frames[r * ncols : (r + 1) * ncols], axis=1))
    return np.concatenate(rows, axis=0)


def _load_adapter(path):
    """Load a wrap_actor function from a Python file."""
    spec = importlib.util.spec_from_file_location("pystk2_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "wrap_actor"):
        raise AttributeError(f"Adapter file {path} must define a wrap_actor function")
    return module.wrap_actor


def _configure_recording(args, env_kwargs):
    """Configure env_kwargs for race recording."""
    import pystk2

    gfx = pystk2.GraphicsConfig.hd()
    if args.hide:
        gfx.display = False
    env_kwargs["graphics_config"] = gfx
    env_kwargs["use_subprocess"] = False
    env_kwargs["num_cameras"] = min(args.cameras or args.num_karts, 8)


_CODEC_MAP = {
    ".mp4": "libx264",
    ".mkv": "libx264",
    ".avi": "mpeg4",
    ".webm": "libvpx",
    ".ogv": "libtheora",
    ".mov": "libx264",
}


def _save_recording(recorded_frames, record_path, fps):
    """Write recorded frames to a video file."""
    if not recorded_frames:
        return
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    path = Path(record_path)
    codec = _CODEC_MAP.get(path.suffix.lower(), "libx264")
    clip = ImageSequenceClip(recorded_frames, fps=fps)
    clip.write_videofile(str(path), codec=codec, logger=None)
    logger.info("Saved %d frames to %s", len(recorded_frames), path)


def _create_actors(loaded_agents, env, wrap_actor):
    """Create actor callables from loaded agents, wrapping errors."""
    actors = []
    for ix, la in enumerate(loaded_agents):
        key = str(ix)
        obs_space = env.observation_space[key]
        act_space = env.action_space[key]
        try:
            actor = la.get_actor(la.state, obs_space, act_space)
            if wrap_actor is not None:
                actor = wrap_actor(actor, obs_space, act_space)
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


def _run_race_inner(args, temp_dirs: list, player_names: list):
    from pystk2_gymnasium.wrappers import MonoAgentWrapperAdapter

    # --- Load agents ---
    loaded_agents: List[LoadedAgent] = []
    for ix, source in enumerate(args.agents):
        logger.info("Loading agent from %s", source)
        try:
            agent = load_agent(source, temp_dirs)
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

    # --- Build agent specs ---
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

    # --- Recording setup ---
    if args.record:
        _configure_recording(args, env_kwargs)

    env = gym.make("supertuxkart/multi-full-v0", **env_kwargs)

    # --- Apply per-agent wrappers ---
    wrapper_factories = {}
    for ix, la in enumerate(loaded_agents):
        wrapper_factories[str(ix)] = _build_wrapper_factory(la)

    env = MonoAgentWrapperAdapter(env, wrapper_factories=wrapper_factories)

    # --- Load adapter if specified ---
    wrap_actor = _load_adapter(args.adapter) if args.adapter else None

    # --- Create actors (deferred: needs obs/action spaces) ---
    actors = _create_actors(loaded_agents, env, wrap_actor)

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
    catch_errors = args.error_handling == "catch"
    action_timeout = args.action_timeout
    max_steps = args.max_steps
    controller = web_server.controller if web_server is not None else None

    recorded_frames = [] if args.record else None
    action_times = [[] for _ in range(num_agents)]

    obs, info = env.reset()
    done = False
    total_rewards = {str(ix): 0.0 for ix in range(num_agents)}
    step_count = 0
    start_time = time.time()

    if web_server is not None:
        web_server.update(env, obs, info, total_rewards, step_count)

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
                    action = _call_with_timeout(actor, (obs[key],), action_timeout)
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

            # Capture frames for recording
            if recorded_frames is not None:
                render_data = env.unwrapped._stk.race.render_data
                cam_images = [np.array(rd.image) for rd in render_data]
                if cam_images:
                    recorded_frames.append(_tile_frames(cam_images))

            # Accumulate per-agent rewards
            agent_rewards = info.get("reward", {})
            for key, r in agent_rewards.items():
                total_rewards[key] = total_rewards.get(key, 0.0) + float(r)

            if web_server is not None:
                web_server.update(env, obs, info, total_rewards, step_count)

    except KeyboardInterrupt:
        logger.info("Race interrupted by user")
    finally:
        elapsed = time.time() - start_time
        env.close()

        if recorded_frames:
            _save_recording(recorded_frames, args.record, args.fps)

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
