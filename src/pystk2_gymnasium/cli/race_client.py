"""Race client: runs the STK environment, dispatches observations to servers.

The client creates the base ``supertuxkart/multi-full-v0`` environment,
sends *base* observations to each server, and receives *base* actions.
All wrapper logic (env_name wrappers + ``get_wrappers()``) is applied
server-side.
"""

import logging
import sys
import time
import traceback
from typing import Dict, List

import gymnasium as gym
import numpy as np

from pystk2_gymnasium.cli.race import (
    FrameRecorder,
    _apply_graphics_config,
    _configure_recording,
    _output_message,
)
from pystk2_gymnasium.cli.race_protocol import (
    MSG_CLOSE,
    MSG_ERROR,
    MSG_INIT,
    MSG_INIT_RESPONSE,
    MSG_SPACES,
    MSG_SPACES_RESPONSE,
    MSG_STEP,
    MSG_STEP_RESPONSE,
    recv_msg,
    send_msg,
)
from pystk2_gymnasium.definitions import AgentException, AgentSpec

logger = logging.getLogger("pystk2.cli.race_client")


class ServerConnection:
    """A connection to a single race server."""

    def __init__(self, address: str, socket):
        self.address = address
        self.socket = socket
        self.agent_keys: List[str] = []  # globally unique keys assigned by client
        self.agents_meta: List[dict] = []  # {player_name, env_name}


def _send_and_recv(
    conn: ServerConnection, msg: dict, expected_type: str, timeout_ms: int
) -> dict:
    """Send a message and receive the expected response, with timeout."""
    send_msg(conn.socket, msg)
    if conn.socket.poll(timeout_ms) == 0:
        raise TimeoutError(
            f"Server {conn.address} did not respond within {timeout_ms / 1000:.0f}s"
        )
    resp = recv_msg(conn.socket)
    if resp["type"] == MSG_ERROR:
        key = resp.get("key")
        tb = resp.get("traceback", "")
        raise AgentException(
            f"Server {conn.address} error: {resp['message']}\n{tb}",
            key,
        )
    if resp["type"] != expected_type:
        raise RuntimeError(
            f"Unexpected response from {conn.address}: "
            f"expected {expected_type}, got {resp['type']}"
        )
    return resp


def _handshake_init(connections, timeout_ms):
    """Send INIT to each server, assign globally unique agent keys.

    Returns the total number of agents.
    """
    global_key_counter = 0
    for conn in connections:
        resp = _send_and_recv(conn, {"type": MSG_INIT}, MSG_INIT_RESPONSE, timeout_ms)
        conn.agents_meta = resp["agents"]
        for meta in conn.agents_meta:
            conn.agent_keys.append(str(global_key_counter))
            if "error" in meta:
                logger.warning(
                    "Server %s: agent %d (%s) failed to load: %s",
                    conn.address,
                    global_key_counter,
                    meta["player_name"],
                    meta["error"],
                )
            global_key_counter += 1
        logger.info(
            "Server %s: %d agent(s) — %s",
            conn.address,
            len(conn.agents_meta),
            [m["player_name"] for m in conn.agents_meta],
        )
    return global_key_counter


def _handshake_spaces(connections, env, timeout_ms):
    """Send base observation/action spaces to each server."""
    for conn in connections:
        obs_spaces = {key: env.observation_space[key] for key in conn.agent_keys}
        act_spaces = {key: env.action_space[key] for key in conn.agent_keys}
        _send_and_recv(
            conn,
            {
                "type": MSG_SPACES,
                "observation_spaces": obs_spaces,
                "action_spaces": act_spaces,
            },
            MSG_SPACES_RESPONSE,
            timeout_ms,
        )
        logger.info("SPACES sent to %s for keys %s", conn.address, conn.agent_keys)


def _collect_actions(
    connections, obs, env, num_agents, catch_errors, action_times, timeout_ms
):
    """Send base observations to all servers and collect base actions."""
    all_actions = {}
    for conn in connections:
        server_obs = {key: obs[key] for key in conn.agent_keys if key in obs}
        resp = _send_and_recv(
            conn,
            {"type": MSG_STEP, "observations": server_obs},
            MSG_STEP_RESPONSE,
            timeout_ms,
        )

        for key, action in resp["actions"].items():
            all_actions[key] = action
        for key, t in resp.get("action_times", {}).items():
            action_times[key].append(t)

        for key, err in resp.get("errors", {}).items():
            if not catch_errors:
                raise AgentException(
                    f"Agent {key} error from {conn.address}: {err['message']}",
                    key,
                )
            logger.warning(
                "Agent %s error from %s: %s — using random action",
                key,
                conn.address,
                err["message"],
            )
            all_actions[key] = env.action_space[key].sample()

    # Fill in any missing actions with random
    for ix in range(num_agents):
        key = str(ix)
        if key not in all_actions:
            all_actions[key] = env.action_space[key].sample()

    return all_actions


def _send_close(connections, timeout_ms):
    """Send CLOSE to all servers, collect per-agent errors.

    Returns a dict ``{agent_key: error_message}`` aggregated from all servers.
    """
    all_errors = {}
    for conn in connections:
        try:
            send_msg(conn.socket, {"type": MSG_CLOSE})
            if conn.socket.poll(timeout_ms):
                resp = recv_msg(conn.socket)
                for key, err in resp.get("errors", {}).items():
                    all_errors[key] = err
        except Exception:
            logger.warning("Failed to send CLOSE to %s", conn.address)
    return all_errors


def _build_results(
    connections,
    info,
    total_rewards,
    action_times,
    env,
    args,
    step_count,
    elapsed,
    server_errors,
):
    """Build and output JSON race results."""
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
    for conn in connections:
        for key, meta in zip(conn.agent_keys, conn.agents_meta):
            agent_info = agent_infos.get(key, {})
            times = action_times.get(key, [])
            avg_action_time = float(np.mean(times)) if times else 0.0
            result_entry = {
                "key": int(key),
                "name": meta["player_name"],
                "reward": rewards.get(key, total_rewards.get(key, 0.0)),
                "position": agent_info.get("position", None),
                "avg_action_time": avg_action_time,
            }
            # Server-reported error (load, init, or runtime failure)
            if key in server_errors:
                result_entry["error"] = server_errors[key]
            results_payload.append(result_entry)
            logger.info(
                "Agent %s (%s): avg action time = %.4fs",
                key,
                meta["player_name"],
                avg_action_time,
            )
    _output_message(message, args)


def run_race_client(args):
    """Run the race client with the given CLI arguments."""
    player_names = []
    try:
        _run_race_client_inner(args, player_names)
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


def _run_race_client_inner(args, player_names: list):
    import zmq

    timeout_ms = int(args.timeout * 1000)
    context = zmq.Context()
    connections: List[ServerConnection] = []

    try:
        for address in args.server:
            sock = context.socket(zmq.REQ)
            sock.connect(address)
            connections.append(ServerConnection(address, sock))
            logger.info("Connected to %s", address)

        # --- Handshake: get agent metadata ---
        num_agents = _handshake_init(connections, timeout_ms)

        agent_specs = []
        for conn in connections:
            for meta in conn.agents_meta:
                agent_specs.append(AgentSpec(name=meta["player_name"]))
                player_names.append(meta["player_name"])

        # --- Create base environment (no wrappers — server applies them) ---
        env = _create_env(args, agent_specs)
        obs, info = env.reset()

        # --- Send base spaces to servers ---
        _handshake_spaces(connections, env, timeout_ms)

        # --- Run race ---
        _run_race_loop(args, env, connections, num_agents, timeout_ms, obs, info)

    finally:
        for conn in connections:
            conn.socket.close()
        context.term()


def _create_env(args, agent_specs):
    """Create the base multi-agent STK environment (no wrappers)."""
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
    _apply_graphics_config(args, env_kwargs)
    if args.record:
        _configure_recording(args, env_kwargs)
    return gym.make("supertuxkart/multi-full-v0", **env_kwargs)


def _run_race_loop(args, env, connections, num_agents, timeout_ms, obs, info):
    """Execute the main race loop, then send CLOSE and output results.

    ``obs`` and ``info`` come from the env.reset() that the caller already
    performed.  The client sends *base* observations; servers handle wrapping.
    """
    from tqdm import tqdm

    # --- Optional web visualization ---
    web_server = None
    if args.web:
        try:
            from pystk2_gymnasium.cli.stk_graph import WebDashboard

            web_server = WebDashboard(
                port=args.web_port,
                num_controlled=num_agents,
                agent_names=[
                    c.agents_meta[i]["player_name"]
                    for c in connections
                    for i in range(len(c.agents_meta))
                ],
            )
            web_server.start()
            logger.info("Web dashboard at http://localhost:%d", args.web_port)
        except ImportError:
            logger.error(
                "Web dashboard requires dash and plotly. "
                "Install with: pip install pystk2-gymnasium[web]"
            )
            sys.exit(1)

    catch_errors = args.error_handling == "catch"
    max_steps = args.max_steps
    max_steps_after_first = getattr(args, "max_steps_after_first", None)
    karts_finished_target = getattr(args, "karts_finished", None)
    controller = web_server.controller if web_server is not None else None
    recorder = None
    if args.record:
        recorder = FrameRecorder(fps=args.fps)
        track_name = getattr(env.unwrapped, "current_track", args.track)
        kart_names = [
            meta["player_name"] for conn in connections for meta in conn.agents_meta
        ]
        recorder.add_title_card(track_name, kart_names)
    action_times: Dict[str, list] = {str(ix): [] for ix in range(num_agents)}
    done = False
    total_rewards = {str(ix): 0.0 for ix in range(num_agents)}
    finished = set()
    step_count = 0
    first_finished_step = None  # step when the first kart finished
    start_time = time.time()

    if web_server is not None:
        web_server.update(env, obs, info, total_rewards, step_count)

    pbar = tqdm(total=max_steps, desc="Racing", unit="step")

    try:
        while not done:
            if controller is not None and not controller.wait_for_step():
                logger.info("Race stopped from web UI")
                break

            all_actions = _collect_actions(
                connections,
                obs,
                env,
                num_agents,
                catch_errors,
                action_times,
                timeout_ms,
            )

            obs, reward, terminated, truncated, info = env.step(all_actions)
            step_count += 1
            done = terminated or truncated
            if max_steps is not None and step_count >= max_steps:
                done = True

            agent_terminated = info.get("terminated", {})
            for key, t in agent_terminated.items():
                if t and key not in finished:
                    finished.add(key)
                    if first_finished_step is None:
                        first_finished_step = step_count

            # Stop after N karts have finished
            if (
                karts_finished_target is not None
                and len(finished) >= karts_finished_target
            ):
                done = True

            # Stop N steps after the first kart finished
            if (
                max_steps_after_first is not None
                and first_finished_step is not None
                and step_count - first_finished_step >= max_steps_after_first
            ):
                done = True

            pbar.update(1)
            pbar.set_postfix(finished=f"{len(finished)}/{num_agents}")

            if recorder is not None:
                screen = env.unwrapped._stk.race.screen_capture()
                if screen is not None and screen.size > 0:
                    recorder.add_frame(np.array(screen))

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
            recorder.save(args.record)
            recorder.cleanup()

    server_errors = _send_close(connections, timeout_ms)
    _build_results(
        connections,
        info,
        total_rewards,
        action_times,
        env,
        args,
        step_count,
        elapsed,
        server_errors,
    )
