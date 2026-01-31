"""Race server: loads agents, creates actors, responds to action requests.

Supports concurrent client sessions via a thread pool (--threads).
Each client gets its own worker thread for the duration of its session.

Wrappers are applied server-side: the client sends *base* observations
(from ``supertuxkart/multi-full-v0``), the server wraps them through each
agent's wrapper chain, calls actors, unwraps the actions, and returns
*base* actions back to the client.
"""

import logging
import os
import pickle
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List

import gymnasium as gym
from gymnasium import spaces

from pystk2_gymnasium.cli.race import (
    LoadedAgent,
    _build_wrapper_factory,
    _call_with_timeout,
    _load_adapter,
    load_agent,
)
from pystk2_gymnasium.definitions import AgentException
from pystk2_gymnasium.cli.race_protocol import (
    MSG_CLOSE,
    MSG_CLOSE_RESPONSE,
    MSG_ERROR,
    MSG_INIT,
    MSG_INIT_RESPONSE,
    MSG_SPACES,
    MSG_SPACES_RESPONSE,
    MSG_STEP,
    MSG_STEP_RESPONSE,
)

logger = logging.getLogger("pystk2.cli.race_server")

# If no message arrives within this many seconds, the session is considered dead.
_SESSION_IDLE_TIMEOUT = 600  # 10 minutes


class _FakeMultiAgentEnv(gym.Env):
    """Minimal env that only carries observation/action spaces.

    Used to construct ``MonoAgentWrapperAdapter`` without a real STK
    environment.  Standard wrappers (``ConstantSizedObservations``,
    ``PolarObservations``, ``FlattenerWrapper``) only inspect the spaces
    during ``__init__``, so this is sufficient.
    """

    def __init__(self, obs_spaces, act_spaces):
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Dict(act_spaces)

    def reset(self, **kwargs):
        raise RuntimeError("_FakeMultiAgentEnv has no real environment")

    def step(self, action):
        raise RuntimeError("_FakeMultiAgentEnv has no real environment")


def _get_msg(msg_queue):
    """Get the next message from the queue, with idle timeout."""
    try:
        return msg_queue.get(timeout=_SESSION_IDLE_TIMEOUT)
    except queue.Empty:
        raise TimeoutError(f"Session idle for {_SESSION_IDLE_TIMEOUT}s, closing")


class _AgentRuntime:
    """Shared wrapper adapter + actors, built once on the first SPACES message.

    Thread-safe: the lock protects the lazy initialization and the
    ``mark_failed`` method so that concurrent sessions don't race.
    Agents that fail (at load, init, or runtime) are added to
    ``failed_keys`` and permanently skipped.
    """

    def __init__(self, loaded_agents, adapter_module):
        self._lock = threading.Lock()
        self._loaded_agents = loaded_agents
        self._adapter_module = adapter_module
        # Set after first init
        self.wrapper_adapter = None
        self.actors = None
        self.failed_keys = None
        self._init_error = None  # (msg, tb) if wrapper building itself failed
        self._key_to_idx = None  # {agent_key: int} mapping
        self._has_state = None  # {agent_key: bool} for stateful agents

    def ensure_initialized(self, agent_keys, obs_spaces, act_spaces):
        """Build wrapper adapter + actors on first call; return cached on later calls.

        Returns ``(error_msg, error_tb)`` on failure, ``None`` on success.
        """
        with self._lock:
            if self._init_error is not None:
                return self._init_error
            if self.actors is not None:
                return None
            return self._build(agent_keys, obs_spaces, act_spaces)

    def mark_failed(self, key, error_msg):
        """Mark an agent as permanently failed (thread-safe)."""
        with self._lock:
            self.failed_keys[key] = error_msg

    def reset_states(self):
        """Return a fresh per-session state dict by calling each agent's reset_state().

        For agents without ``reset_state``, the value is ``None``.
        """
        states = {}
        for key, has in self._has_state.items():
            if has:
                idx = self._key_to_idx[key]
                states[key] = self._loaded_agents[idx].reset_state()
            else:
                states[key] = None
        return states

    def _build(self, agent_keys, obs_spaces, act_spaces):
        key_to_idx = {k: i for i, k in enumerate(agent_keys)}
        self._key_to_idx = key_to_idx

        try:
            self.wrapper_adapter = _build_wrapper_adapter(
                agent_keys, key_to_idx, self._loaded_agents, obs_spaces, act_spaces
            )
        except Exception as exc:
            self._init_error = (
                f"Failed to build wrappers: {exc}",
                traceback.format_exc(),
            )
            return self._init_error

        create_actor = (
            getattr(self._adapter_module, "create_actor", None)
            if self._adapter_module
            else None
        )
        actors = {}
        failed_keys = {}
        for key in agent_keys:
            idx = key_to_idx[key]
            la = self._loaded_agents[idx]

            if la.load_error:
                failed_keys[key] = la.load_error
                logger.warning(
                    "Skipping actor for key %s (agent %d: %s): %s",
                    key,
                    idx,
                    la.player_name,
                    la.load_error,
                )
                continue

            wrapped_obs_space = self.wrapper_adapter.observation_space[key]
            wrapped_act_space = self.wrapper_adapter.action_space[key]
            try:
                if create_actor is not None:
                    actor = create_actor(
                        la.get_actor,
                        la.module_dir,
                        wrapped_obs_space,
                        wrapped_act_space,
                    )
                else:
                    actor = la.get_actor(
                        la.module_dir, wrapped_obs_space, wrapped_act_space
                    )
            except Exception as exc:
                error_msg = (
                    f"Failed to create actor for agent {idx} ({la.player_name}): {exc}"
                )
                failed_keys[key] = error_msg
                logger.error("%s\n%s", error_msg, traceback.format_exc())
                continue
            actors[key] = actor
            logger.info(
                "Created actor for key %s (agent %d: %s)",
                key,
                idx,
                la.player_name,
            )

        self.actors = actors
        self.failed_keys = failed_keys
        self._has_state = {
            k: self._loaded_agents[key_to_idx[k]].reset_state is not None
            for k in agent_keys
        }
        return None


def _session_worker(
    msg_queue,
    send_fn,
    cleanup_fn,
    loaded_agents,
    args,
    runtime,
    session_id,
):
    """Handle one client session in a worker thread."""
    logger.info("Session %s: started", session_id)
    try:
        _handle_session(msg_queue, send_fn, loaded_agents, args, runtime)
    except TimeoutError as e:
        logger.warning("Session %s: %s", session_id, e)
    except Exception:
        logger.exception("Session %s: error", session_id)
        try:
            send_fn(
                {
                    "type": MSG_ERROR,
                    "message": "Internal server error",
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        logger.info("Session %s: ended", session_id)
        cleanup_fn()


def _build_wrapper_adapter(
    agent_keys, key_to_idx, loaded_agents, obs_spaces, act_spaces
):
    """Build a MonoAgentWrapperAdapter on a fake env for server-side wrapping.

    Agents with ``load_error`` are skipped (no wrappers applied for them).
    """
    from pystk2_gymnasium.wrappers import MonoAgentWrapperAdapter

    fake_env = _FakeMultiAgentEnv(obs_spaces, act_spaces)
    wrapper_factories = {}
    for key in agent_keys:
        idx = key_to_idx[key]
        la = loaded_agents[idx]
        if la.load_error:
            continue
        wrapper_factories[key] = _build_wrapper_factory(la)

    return MonoAgentWrapperAdapter(fake_env, wrapper_factories=wrapper_factories)


def _handle_session(msg_queue, send_fn, loaded_agents, args, runtime):
    """Handle one client session (INIT -> SPACES -> STEP* -> CLOSE).

    Wrappers and actors are built once (on the first session) and shared
    across all sessions via *runtime*.  The client sends *base*
    observations and receives *base* actions.

    Any per-agent error (load, init, wrapper, actor) marks the agent as
    permanently failed.  The CLOSE response includes all errors so the
    client can build its race report.
    """
    action_timeout = getattr(args, "action_timeout", None)

    # --- Wait for INIT ---
    msg = _get_msg(msg_queue)
    if msg["type"] != MSG_INIT:
        send_fn({"type": MSG_ERROR, "message": f"Expected INIT, got {msg['type']}"})
        return

    agents_meta = []
    for la in loaded_agents:
        meta = {"player_name": la.player_name, "env_name": la.env_name}
        if la.load_error:
            meta["error"] = la.load_error
        agents_meta.append(meta)
    send_fn({"type": MSG_INIT_RESPONSE, "agents": agents_meta})
    logger.info("INIT: sent metadata for %d agent(s)", len(loaded_agents))

    # --- Wait for SPACES ---
    msg = _get_msg(msg_queue)
    if msg["type"] != MSG_SPACES:
        send_fn({"type": MSG_ERROR, "message": f"Expected SPACES, got {msg['type']}"})
        return

    # Client sends the *base* spaces (from supertuxkart/multi-full-v0)
    obs_spaces = msg["observation_spaces"]  # {agent_key: base Space}
    act_spaces = msg["action_spaces"]  # {agent_key: base Space}
    agent_keys = sorted(obs_spaces.keys())
    logger.info("SPACES: received base spaces for keys %s", agent_keys)

    if len(agent_keys) != len(loaded_agents):
        send_fn(
            {
                "type": MSG_ERROR,
                "message": (
                    f"Server has {len(loaded_agents)} agent(s) but "
                    f"received spaces for {len(agent_keys)}"
                ),
            }
        )
        return

    key_to_idx = {k: i for i, k in enumerate(agent_keys)}

    # Build wrappers + actors on the first session; reuse on subsequent ones.
    init_error = runtime.ensure_initialized(agent_keys, obs_spaces, act_spaces)
    if init_error is not None:
        msg_text, tb = init_error
        send_fn({"type": MSG_ERROR, "message": msg_text, "traceback": tb})
        return

    wrapper_adapter = runtime.wrapper_adapter
    actors = runtime.actors
    failed_keys = runtime.failed_keys

    # Initialize per-session state for stateful agents
    states = runtime.reset_states()

    send_fn({"type": MSG_SPACES_RESPONSE, "status": "ok"})

    # --- Step loop ---
    while True:
        msg = _get_msg(msg_queue)

        if msg["type"] == MSG_CLOSE:
            # Include accumulated errors so the client can build its report
            send_fn(
                {
                    "type": MSG_CLOSE_RESPONSE,
                    "status": "ok",
                    "errors": dict(failed_keys),
                }
            )
            logger.info("CLOSE: session ended")
            return

        if msg["type"] != MSG_STEP:
            send_fn(
                {
                    "type": MSG_ERROR,
                    "message": f"Expected STEP or CLOSE, got {msg['type']}",
                }
            )
            return

        # Client sends *base* observations; wrap → actor → unwrap
        base_observations = msg["observations"]

        # Wrap base observations, call actors, unwrap actions — per agent
        wrapped_actions = {}
        action_times = {}
        errors = {}

        for key in agent_keys:
            if key not in base_observations or key in failed_keys:
                continue
            idx = key_to_idx[key]
            la = loaded_agents[idx]

            # Wrap observation
            try:
                wrapped_obs = wrapper_adapter.observation({key: base_observations[key]})
            except AgentException as exc:
                error_msg = f"Wrapper observation error: {exc}"
                logger.warning(
                    "Agent %d (%s, key=%s): %s", idx, la.player_name, key, error_msg
                )
                runtime.mark_failed(key, error_msg)
                errors[key] = {"message": error_msg}
                continue

            # Call actor
            try:
                t_start = time.perf_counter()
                if states.get(key) is not None:
                    actor_args = (states[key], wrapped_obs[key])
                else:
                    actor_args = (wrapped_obs[key],)
                action = _call_with_timeout(actors[key], actor_args, action_timeout)
                action_times[key] = time.perf_counter() - t_start
            except Exception as exc:
                error_msg = f"Actor error: {exc}"
                logger.warning(
                    "Agent %d (%s, key=%s): %s", idx, la.player_name, key, error_msg
                )
                runtime.mark_failed(key, error_msg)
                errors[key] = {"message": error_msg}
                continue

            # Unwrap action
            try:
                base_action = wrapper_adapter.action({key: action})
                wrapped_actions.update(base_action)
            except AgentException as exc:
                error_msg = f"Wrapper action error: {exc}"
                logger.warning(
                    "Agent %d (%s, key=%s): %s", idx, la.player_name, key, error_msg
                )
                runtime.mark_failed(key, error_msg)
                errors[key] = {"message": error_msg}

        base_actions = wrapped_actions

        send_fn(
            {
                "type": MSG_STEP_RESPONSE,
                "actions": base_actions,
                "action_times": action_times,
                "errors": errors,
            }
        )


def run_race_server(args):
    """Run the race server with a thread pool for concurrent sessions."""
    import zmq

    num_threads = getattr(args, "threads", None)
    if num_threads is None:
        num_threads = max(1, (os.cpu_count() or 2) // 2)

    # --- Load agents (shared across all sessions, read-only) ---
    temp_dirs = []
    context = None
    socket = None
    try:
        adapter_module = _load_adapter(args.adapter) if args.adapter else None
        prepare_module_dir = getattr(adapter_module, "prepare_module_dir", None)

        loaded_agents: List[LoadedAgent] = []
        for source in args.agents:
            logger.info("Loading agent from %s", source)
            try:
                agent = load_agent(
                    source, temp_dirs, prepare_module_dir=prepare_module_dir
                )
                logger.info(
                    "Loaded agent %r (env=%s) from %s",
                    agent.player_name,
                    agent.env_name,
                    source,
                )
            except Exception as exc:
                error_msg = f"Failed to load agent from {source}: {exc}"
                logger.error("%s\n%s", error_msg, traceback.format_exc())
                # Parse name override if present
                name = source.rsplit("@:", 1)[1] if "@:" in source else source
                agent = LoadedAgent(
                    env_name="supertuxkart/full-v0",
                    player_name=name,
                    get_actor=None,
                    module_dir=None,
                    get_wrappers=None,
                    source=source,
                    load_error=error_msg,
                )
            loaded_agents.append(agent)

        # Shared runtime: wrappers + actors built once on first session
        runtime = _AgentRuntime(loaded_agents, adapter_module)

        logger.info(
            "Server ready with %d agent(s), %d thread(s): %s",
            len(loaded_agents),
            num_threads,
            [la.player_name for la in loaded_agents],
        )

        # --- ZMQ setup: ROUTER socket for concurrent clients ---
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.bind(args.address)
        logger.info("Server bound on %s", args.address)

        executor = ThreadPoolExecutor(max_workers=num_threads)
        client_queues = {}  # identity bytes -> Queue
        clients_lock = threading.Lock()
        send_lock = threading.Lock()
        session_counter = 0

        def _make_send_fn(identity):
            """Create a thread-safe send function for a specific client."""

            def send_fn(msg_dict):
                with send_lock:
                    socket.send_multipart([identity, b"", pickle.dumps(msg_dict)])

            return send_fn

        def _make_cleanup_fn(identity):
            """Create a cleanup function that removes the client from routing."""

            def cleanup_fn():
                with clients_lock:
                    client_queues.pop(identity, None)

            return cleanup_fn

        # --- Main recv loop: dispatch messages to worker threads ---
        while True:
            frames = socket.recv_multipart()
            identity = frames[0]
            data = frames[-1]
            msg = pickle.loads(data)

            with clients_lock:
                if identity not in client_queues:
                    session_counter += 1
                    q = queue.Queue()
                    client_queues[identity] = q
                    executor.submit(
                        _session_worker,
                        q,
                        _make_send_fn(identity),
                        _make_cleanup_fn(identity),
                        loaded_agents,
                        args,
                        runtime,
                        session_counter,
                    )
                client_queues[identity].put(msg)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        for td in temp_dirs:
            td.cleanup()
        if socket is not None:
            socket.close()
        if context is not None:
            context.term()
