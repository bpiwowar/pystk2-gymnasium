"""Race command: load agents, run a race, output results."""

import importlib
import json
import logging
import signal
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import gymnasium as gym

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
    """Load pystk_actor module from a directory path."""
    actor_path = path / "pystk_actor.py"
    if not actor_path.exists():
        raise FileNotFoundError(f"No pystk_actor.py found in {path}")

    import importlib.util

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

    elif path.is_dir():
        module = _load_module_from_path(path)
        state = _load_weights(path)

    else:
        # Try as Python module name
        try:
            full_module = f"{source}.pystk_actor"
            module = importlib.import_module(full_module)
        except ImportError:
            # Maybe the source itself is the module
            module = importlib.import_module(source)
        state = None

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

    # Add extra wrappers from the agent
    if loaded.get_wrappers is not None:
        extra = loaded.get_wrappers()
        if extra:
            wrapper_specs.extend(extra)

    def factory(env):
        wrapped = env
        for ws in wrapper_specs:
            wrapper_cls = load_env_creator(ws.entry_point)
            wrapped = wrapper_cls(wrapped, **(ws.kwargs or {}))
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


def run_race(args):
    """Run a race with the given CLI arguments."""
    temp_dirs = []
    try:
        _run_race_inner(args, temp_dirs)
    finally:
        for td in temp_dirs:
            td.cleanup()


def _run_race_inner(args, temp_dirs: list):
    from pystk2_gymnasium.wrappers import MonoAgentWrapperAdapter

    # --- Load agents ---
    loaded_agents: List[LoadedAgent] = []
    for source in args.agents:
        logger.info("Loading agent from %s", source)
        agent = load_agent(source, temp_dirs)
        logger.info(
            "Loaded agent %r (env=%s) from %s",
            agent.player_name,
            agent.env_name,
            source,
        )
        loaded_agents.append(agent)

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

    env = gym.make("supertuxkart/multi-full-v0", **env_kwargs)

    # --- Apply per-agent wrappers ---
    wrapper_factories = {}
    for ix, la in enumerate(loaded_agents):
        wrapper_factories[str(ix)] = _build_wrapper_factory(la)

    env = MonoAgentWrapperAdapter(env, wrapper_factories=wrapper_factories)

    # --- Create actors (deferred: needs obs/action spaces) ---
    actors = []
    for ix, la in enumerate(loaded_agents):
        key = str(ix)
        obs_space = env.observation_space[key]
        act_space = env.action_space[key]
        actor = la.get_actor(la.state, obs_space, act_space)
        actors.append(actor)
        logger.info("Created actor for agent %d (%s)", ix, la.player_name)

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
    controller = web_server.controller if web_server is not None else None

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
                    action = _call_with_timeout(actor, (obs[key],), action_timeout)
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

    # --- Build results ---
    agent_infos = info.get("infos", {})
    results = {
        "track": getattr(env.unwrapped, "current_track", args.track),
        "steps": step_count,
        "elapsed_seconds": round(elapsed, 2),
        "agents": [],
    }
    for ix, la in enumerate(loaded_agents):
        key = str(ix)
        agent_info = agent_infos.get(key, {})
        results["agents"].append(
            {
                "index": ix,
                "name": la.player_name,
                "source": la.source,
                "env_name": la.env_name,
                "total_reward": round(total_rewards.get(key, 0.0), 4),
                "position": agent_info.get("position", None),
                "distance": agent_info.get("distance", None),
            }
        )

    # --- Output results ---
    results_json = json.dumps(results, indent=2)
    logger.info("Race results:\n%s", results_json)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(results_json)
        logger.info("Results written to %s", output_path)
