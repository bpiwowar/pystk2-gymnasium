"""Tests for unified stateful agent interface (create_state / state passing).

Covers:
- Agent loading (create_state captured from module)
- Local race state initialization and passing
- Server-side _AgentRuntime state management
- Full client-server protocol with stateful agents
- Mixed stateful / stateless agents
- Per-session state isolation on the server
"""

import pickle
import queue
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gymnasium import spaces

from pystk2_gymnasium.cli.race import (
    LoadedAgent,
    _call_with_timeout,
    load_agent,
)
from pystk2_gymnasium.cli.race_protocol import (
    MSG_CLOSE,
    MSG_CLOSE_RESPONSE,
    MSG_INIT,
    MSG_INIT_RESPONSE,
    MSG_SPACES,
    MSG_SPACES_RESPONSE,
    MSG_STEP,
    MSG_STEP_RESPONSE,
)
from pystk2_gymnasium.cli.race_server import (
    _AgentRuntime,
    _FakeMultiAgentEnv,
    _handle_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_obs_space():
    """A minimal observation space for testing."""
    return spaces.Dict({"x": spaces.Box(-1, 1, shape=(2,), dtype=np.float32)})


def _simple_act_space():
    """A minimal action space for testing."""
    return spaces.Dict({"acceleration": spaces.Box(0, 1, shape=(1,), dtype=np.float32)})


def _make_loaded_agent(
    name="test",
    stateful=False,
    state_factory=None,
    actor_fn=None,
):
    """Build a LoadedAgent for testing.

    All actors use the unified ``(state, obs)`` signature.
    If *stateful*, ``create_state`` returns a dict; otherwise it returns ``None``.
    *state_factory* overrides the default create_state callable.
    *actor_fn* overrides the default get_actor callable.
    """

    def default_stateless_actor(module_dir, obs_space, act_space):
        def actor(state, obs):
            return {"acceleration": np.array([0.5], dtype=np.float32)}

        return actor

    def default_stateful_actor(module_dir, obs_space, act_space):
        def actor(state, obs):
            state["counter"] += 1
            return {"acceleration": np.array([0.5], dtype=np.float32)}

        return actor

    if stateful:
        create_state = state_factory or (lambda: {"counter": 0})
        get_actor = actor_fn or default_stateful_actor
    else:
        create_state = state_factory or (lambda: None)
        get_actor = actor_fn or default_stateless_actor

    return LoadedAgent(
        env_name="supertuxkart/full-v0",
        player_name=name,
        get_actor=get_actor,
        module_dir=Path("."),
        get_wrappers=None,
        source=name,
        create_state=create_state,
    )


def _make_fake_env(n_agents):
    """Create a _FakeMultiAgentEnv with n_agents."""
    obs_spaces = {str(i): _simple_obs_space() for i in range(n_agents)}
    act_spaces = {str(i): _simple_act_space() for i in range(n_agents)}
    return _FakeMultiAgentEnv(obs_spaces, act_spaces)


# ---------------------------------------------------------------------------
# 1. Agent loading
# ---------------------------------------------------------------------------


class TestLoadAgent:
    """Test that load_agent() captures create_state from agent modules."""

    def test_stateless_agent_default_create_state(self, tmp_path):
        """Agent without create_state gets default lambda returning None."""
        agent_dir = tmp_path / "agent_stateless"
        agent_dir.mkdir()
        (agent_dir / "pystk_actor.py").write_text(
            textwrap.dedent("""\
                def get_actor(state, obs_space, act_space):
                    return lambda state, obs: {}
            """)
        )
        la = load_agent(str(agent_dir), [])
        assert la.create_state() is None

    def test_stateful_agent_has_create_state(self, tmp_path):
        """Agent with create_state gets captured."""
        agent_dir = tmp_path / "agent_stateful"
        agent_dir.mkdir()
        (agent_dir / "pystk_actor.py").write_text(
            textwrap.dedent("""\
                def create_state():
                    return {"step_count": 0}

                def get_actor(state, obs_space, act_space):
                    def act(agent_state, obs):
                        agent_state["step_count"] += 1
                        return {}
                    return act
            """)
        )
        la = load_agent(str(agent_dir), [])
        state = la.create_state()
        assert state == {"step_count": 0}

    def test_create_state_returns_fresh_state(self, tmp_path):
        """Each call to create_state returns an independent object."""
        agent_dir = tmp_path / "agent_fresh"
        agent_dir.mkdir()
        (agent_dir / "pystk_actor.py").write_text(
            textwrap.dedent("""\
                def create_state():
                    return {"n": 0}

                def get_actor(state, obs_space, act_space):
                    return lambda s, obs: {}
            """)
        )
        la = load_agent(str(agent_dir), [])
        s1 = la.create_state()
        s1["n"] = 42
        s2 = la.create_state()
        assert s2["n"] == 0, "create_state must return a fresh object each time"


# ---------------------------------------------------------------------------
# 2. LoadedAgent dataclass
# ---------------------------------------------------------------------------


class TestLoadedAgentDataclass:
    def test_default_create_state_returns_none(self):
        la = LoadedAgent(
            env_name="e",
            player_name="p",
            get_actor=lambda *a: None,
            module_dir=Path("."),
            get_wrappers=None,
            source="s",
        )
        assert la.create_state() is None

    def test_create_state_field(self):
        fn = lambda: {"x": 1}  # noqa: E731
        la = LoadedAgent(
            env_name="e",
            player_name="p",
            get_actor=lambda *a: None,
            module_dir=Path("."),
            get_wrappers=None,
            source="s",
            create_state=fn,
        )
        assert la.create_state is fn
        assert la.create_state() == {"x": 1}


# ---------------------------------------------------------------------------
# 3. Local race state management
# ---------------------------------------------------------------------------


class TestLocalRaceState:
    """Test state initialization and passing in the local-race code path."""

    def test_states_initialized_from_loaded_agents(self):
        """States list built correctly for mixed agents."""
        agents = [
            _make_loaded_agent("stateless", stateful=False),
            _make_loaded_agent("stateful", stateful=True),
            _make_loaded_agent("stateless2", stateful=False),
        ]
        states = [la.create_state() for la in agents]
        assert states[0] is None
        assert states[1] == {"counter": 0}
        assert states[2] is None

    def test_actor_always_receives_state_and_obs(self):
        """Actor is always called with (state, obs), even when state is None."""
        call_log = []

        def my_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                call_log.append(("called", state, obs))
                return {"acceleration": np.array([1.0], dtype=np.float32)}

            return actor

        la = _make_loaded_agent("s", stateful=True, actor_fn=my_get_actor)
        actor = la.get_actor(la.module_dir, _simple_obs_space(), _simple_act_space())
        state = la.create_state()
        obs = {"x": np.array([0.1, 0.2], dtype=np.float32)}

        # Simulate what _run_race_inner does: always pass (state, obs)
        _call_with_timeout(actor, (state, obs), None)

        assert len(call_log) == 1
        assert call_log[0][1] is state

    def test_stateless_actor_receives_none_state(self):
        """Stateless actor is called with (None, obs)."""
        call_log = []

        def my_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                call_log.append(("stateless", state, obs))
                return {"acceleration": np.array([1.0], dtype=np.float32)}

            return actor

        la = _make_loaded_agent("s", stateful=False, actor_fn=my_get_actor)
        actor = la.get_actor(la.module_dir, _simple_obs_space(), _simple_act_space())
        state = la.create_state()
        obs = {"x": np.array([0.1, 0.2], dtype=np.float32)}

        _call_with_timeout(actor, (state, obs), None)

        assert len(call_log) == 1
        assert call_log[0][0] == "stateless"
        assert call_log[0][1] is None

    def test_state_mutated_across_steps(self):
        """Stateful actor can mutate state across multiple steps."""
        la = _make_loaded_agent("counter", stateful=True)
        actor = la.get_actor(la.module_dir, _simple_obs_space(), _simple_act_space())
        state = la.create_state()
        obs = {"x": np.array([0.0, 0.0], dtype=np.float32)}

        for _ in range(5):
            _call_with_timeout(actor, (state, obs), None)

        assert state["counter"] == 5

    def test_create_state_resets_counter(self):
        """Calling create_state after mutations returns a fresh state."""
        la = _make_loaded_agent("counter", stateful=True)
        actor = la.get_actor(la.module_dir, _simple_obs_space(), _simple_act_space())
        state = la.create_state()
        obs = {"x": np.array([0.0, 0.0], dtype=np.float32)}

        for _ in range(3):
            _call_with_timeout(actor, (state, obs), None)
        assert state["counter"] == 3

        # Simulate a race reset
        state2 = la.create_state()
        assert state2["counter"] == 0
        assert state["counter"] == 3  # original unchanged


# ---------------------------------------------------------------------------
# 4. _AgentRuntime state management (server-side)
# ---------------------------------------------------------------------------


class TestAgentRuntime:
    """Test _AgentRuntime initialization and reset_states()."""

    def _build_runtime(self, agents):
        """Helper: build and initialize a runtime."""
        runtime = _AgentRuntime(agents, adapter_module=None)
        keys = [str(i) for i in range(len(agents))]
        obs_spaces = {k: _simple_obs_space() for k in keys}
        act_spaces = {k: _simple_act_space() for k in keys}
        err = runtime.ensure_initialized(keys, obs_spaces, act_spaces)
        assert err is None, f"Runtime init failed: {err}"
        return runtime, keys

    def test_reset_states_all_stateless(self):
        agents = [
            _make_loaded_agent("a", stateful=False),
            _make_loaded_agent("b", stateful=False),
        ]
        runtime, keys = self._build_runtime(agents)
        states = runtime.reset_states()
        assert states["0"] is None
        assert states["1"] is None

    def test_reset_states_all_stateful(self):
        agents = [
            _make_loaded_agent("a", stateful=True),
            _make_loaded_agent("b", stateful=True),
        ]
        runtime, keys = self._build_runtime(agents)
        states = runtime.reset_states()
        assert states["0"] == {"counter": 0}
        assert states["1"] == {"counter": 0}

    def test_reset_states_mixed(self):
        agents = [
            _make_loaded_agent("stateless", stateful=False),
            _make_loaded_agent("stateful", stateful=True),
        ]
        runtime, keys = self._build_runtime(agents)
        states = runtime.reset_states()
        assert states["0"] is None
        assert states["1"] == {"counter": 0}

    def test_reset_states_returns_fresh_per_call(self):
        """Each call to reset_states() returns independent state objects."""
        agents = [_make_loaded_agent("s", stateful=True)]
        runtime, keys = self._build_runtime(agents)

        s1 = runtime.reset_states()
        s1["0"]["counter"] = 99
        s2 = runtime.reset_states()
        assert s2["0"]["counter"] == 0

    def test_key_to_idx_stored(self):
        agents = [
            _make_loaded_agent("a", stateful=False),
            _make_loaded_agent("b", stateful=True),
        ]
        runtime, keys = self._build_runtime(agents)
        assert runtime._key_to_idx == {"0": 0, "1": 1}


# ---------------------------------------------------------------------------
# 5. Full server session with stateful agents
# ---------------------------------------------------------------------------


class TestServerSession:
    """Test _handle_session with stateful agents via message queues.

    Uses _FakeMultiAgentEnv so no real STK instance is needed.
    """

    def _run_session(self, agents, steps=3):
        """Run a server session through its full protocol, return responses.

        Patches MonoAgentWrapperAdapter to be a pass-through (identity wrapper).
        """
        runtime = _AgentRuntime(agents, adapter_module=None)
        n = len(agents)
        keys = [str(i) for i in range(n)]

        obs_space = _simple_obs_space()
        act_space = _simple_act_space()
        obs_spaces = {k: obs_space for k in keys}
        act_spaces = {k: act_space for k in keys}

        # Collect sent responses
        responses = []

        def send_fn(msg):
            responses.append(msg)

        msg_queue = queue.Queue()

        # Feed messages: INIT -> SPACES -> STEP*N -> CLOSE
        msg_queue.put({"type": MSG_INIT})
        msg_queue.put(
            {
                "type": MSG_SPACES,
                "observation_spaces": obs_spaces,
                "action_spaces": act_spaces,
            }
        )
        for _ in range(steps):
            obs = {k: {"x": np.array([0.1, 0.2], dtype=np.float32)} for k in keys}
            msg_queue.put({"type": MSG_STEP, "observations": obs})
        msg_queue.put({"type": MSG_CLOSE})

        args = SimpleNamespace(action_timeout=None)

        # Patch MonoAgentWrapperAdapter to pass through
        with patch(
            "pystk2_gymnasium.cli.race_server._build_wrapper_adapter"
        ) as mock_bwa:
            # Create a mock wrapper adapter that passes observations/actions through
            mock_adapter = MagicMock()
            mock_adapter.observation_space = obs_spaces
            mock_adapter.action_space = act_spaces
            mock_adapter.observation.side_effect = lambda obs: obs
            mock_adapter.action.side_effect = lambda acts: acts
            mock_bwa.return_value = mock_adapter

            _handle_session(msg_queue, send_fn, agents, args, runtime)

        return responses

    def test_stateless_agent_session(self):
        agents = [_make_loaded_agent("sl", stateful=False)]
        responses = self._run_session(agents, steps=3)

        # INIT_RESPONSE, SPACES_RESPONSE, 3x STEP_RESPONSE, CLOSE_RESPONSE
        assert responses[0]["type"] == MSG_INIT_RESPONSE
        assert responses[1]["type"] == MSG_SPACES_RESPONSE
        for i in range(2, 5):
            assert responses[i]["type"] == MSG_STEP_RESPONSE
            assert "0" in responses[i]["actions"]
        assert responses[5]["type"] == MSG_CLOSE_RESPONSE

    def test_stateful_agent_session(self):
        agents = [_make_loaded_agent("sf", stateful=True)]
        responses = self._run_session(agents, steps=3)

        assert responses[0]["type"] == MSG_INIT_RESPONSE
        assert responses[1]["type"] == MSG_SPACES_RESPONSE
        for i in range(2, 5):
            assert responses[i]["type"] == MSG_STEP_RESPONSE
            assert "0" in responses[i]["actions"]
        assert responses[5]["type"] == MSG_CLOSE_RESPONSE

    def test_mixed_agents_session(self):
        agents = [
            _make_loaded_agent("stateless", stateful=False),
            _make_loaded_agent("stateful", stateful=True),
        ]
        responses = self._run_session(agents, steps=2)

        assert responses[0]["type"] == MSG_INIT_RESPONSE
        assert len(responses[0]["agents"]) == 2
        assert responses[1]["type"] == MSG_SPACES_RESPONSE
        # 2 steps
        for i in range(2, 4):
            r = responses[i]
            assert r["type"] == MSG_STEP_RESPONSE
            assert "0" in r["actions"]
            assert "1" in r["actions"]
        assert responses[4]["type"] == MSG_CLOSE_RESPONSE

    def test_stateful_agent_state_mutates_across_steps(self):
        """Verify that the stateful actor's state accumulates across steps."""
        # Track calls: actor records state["counter"] at each step
        call_counters = []

        def tracking_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                call_counters.append(state["counter"])
                state["counter"] += 1
                return {"acceleration": np.array([0.5], dtype=np.float32)}

            return actor

        agents = [
            _make_loaded_agent(
                "tracker",
                stateful=True,
                actor_fn=tracking_get_actor,
            )
        ]
        self._run_session(agents, steps=4)

        # State starts at 0 and increments each step
        assert call_counters == [0, 1, 2, 3]

    def test_per_session_state_isolation(self):
        """Two sequential sessions get independent state."""
        call_counters_session1 = []
        call_counters_session2 = []

        # We need a fresh list per session to track separately
        session_call_log = []

        def tracking_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                session_call_log.append(state["counter"])
                state["counter"] += 1
                return {"acceleration": np.array([0.5], dtype=np.float32)}

            return actor

        agents = [
            _make_loaded_agent(
                "tracker",
                stateful=True,
                actor_fn=tracking_get_actor,
            )
        ]

        # Session 1
        session_call_log.clear()
        self._run_session(agents, steps=3)
        call_counters_session1 = list(session_call_log)

        # Session 2 (reuses same runtime via _run_session creating a new one)
        session_call_log.clear()
        self._run_session(agents, steps=3)
        call_counters_session2 = list(session_call_log)

        # Both sessions should start from 0
        assert call_counters_session1 == [0, 1, 2]
        assert call_counters_session2 == [0, 1, 2]


# ---------------------------------------------------------------------------
# 6. Full ZMQ client-server integration
# ---------------------------------------------------------------------------


class TestZMQIntegration:
    """Integration tests using real ZMQ sockets (inproc transport).

    These test the full protocol from the server side: message serialization,
    state management, and action responses.
    """

    @pytest.fixture
    def zmq_context(self):
        zmq = pytest.importorskip("zmq")
        ctx = zmq.Context()
        yield ctx
        ctx.term()

    def _start_server_thread(self, zmq_context, address, agents, adapter=None):
        """Start a minimal server loop in a background thread.

        Returns (thread, stop_event) — set stop_event to terminate.
        """
        import zmq

        runtime = _AgentRuntime(agents, adapter)
        args = SimpleNamespace(action_timeout=None)

        def server_loop():
            socket = zmq_context.socket(zmq.ROUTER)
            socket.bind(address)
            # Single-session: one client
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)

            msg_queue = queue.Queue()
            responses = []
            session_done = threading.Event()

            def send_fn(msg_dict):
                # send_fn is called from the session handler thread
                responses.append(msg_dict)

            def handle():
                with patch(
                    "pystk2_gymnasium.cli.race_server._build_wrapper_adapter"
                ) as mock_bwa:
                    obs_space = _simple_obs_space()
                    act_space = _simple_act_space()
                    n = len(agents)
                    keys = [str(i) for i in range(n)]
                    obs_spaces = {k: obs_space for k in keys}
                    act_spaces = {k: act_space for k in keys}

                    mock_adapter = MagicMock()
                    mock_adapter.observation_space = obs_spaces
                    mock_adapter.action_space = act_spaces
                    mock_adapter.observation.side_effect = lambda obs: obs
                    mock_adapter.action.side_effect = lambda acts: acts
                    mock_bwa.return_value = mock_adapter

                    _handle_session(msg_queue, send_fn, agents, args, runtime)
                session_done.set()

            handler_thread = threading.Thread(target=handle, daemon=True)
            handler_thread.start()

            identity = None
            while not session_done.is_set():
                # Poll for incoming messages
                socks = dict(poller.poll(100))
                if socket in socks:
                    frames = socket.recv_multipart()
                    identity = frames[0]
                    data = frames[-1]
                    msg = pickle.loads(data)
                    msg_queue.put(msg)

                # Send any pending responses
                while responses:
                    resp = responses.pop(0)
                    socket.send_multipart([identity, b"", pickle.dumps(resp)])

            handler_thread.join(timeout=5)
            socket.close()

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
        return thread

    def test_zmq_stateful_session(self, zmq_context):
        """Full ZMQ round-trip with a stateful agent."""
        zmq = pytest.importorskip("zmq")
        from pystk2_gymnasium.cli.race_protocol import recv_msg, send_msg

        address = "tcp://127.0.0.1:15566"

        call_log = []

        def tracking_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                call_log.append(state["counter"])
                state["counter"] += 1
                return {"acceleration": np.array([0.5], dtype=np.float32)}

            return actor

        agents = [
            _make_loaded_agent("zmq_sf", stateful=True, actor_fn=tracking_get_actor)
        ]
        thread = self._start_server_thread(zmq_context, address, agents)
        time.sleep(0.2)  # let server bind

        # Client side
        client = zmq_context.socket(zmq.REQ)
        client.connect(address)

        try:
            # INIT
            send_msg(client, {"type": MSG_INIT})
            resp = recv_msg(client)
            assert resp["type"] == MSG_INIT_RESPONSE
            assert len(resp["agents"]) == 1

            # SPACES
            obs_space = _simple_obs_space()
            act_space = _simple_act_space()
            send_msg(
                client,
                {
                    "type": MSG_SPACES,
                    "observation_spaces": {"0": obs_space},
                    "action_spaces": {"0": act_space},
                },
            )
            resp = recv_msg(client)
            assert resp["type"] == MSG_SPACES_RESPONSE

            # STEP x3
            for i in range(3):
                obs = {"0": {"x": np.array([0.1, 0.2], dtype=np.float32)}}
                send_msg(client, {"type": MSG_STEP, "observations": obs})
                resp = recv_msg(client)
                assert resp["type"] == MSG_STEP_RESPONSE
                assert "0" in resp["actions"]

            # CLOSE
            send_msg(client, {"type": MSG_CLOSE})
            resp = recv_msg(client)
            assert resp["type"] == MSG_CLOSE_RESPONSE

            # Verify state was accumulated across steps
            assert call_log == [0, 1, 2]

        finally:
            client.close()
            thread.join(timeout=5)

    def test_zmq_stateless_session(self, zmq_context):
        """Full ZMQ round-trip with a stateless agent."""
        zmq = pytest.importorskip("zmq")
        from pystk2_gymnasium.cli.race_protocol import recv_msg, send_msg

        address = "tcp://127.0.0.1:15567"

        call_log = []

        def tracking_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                call_log.append("called")
                return {"acceleration": np.array([0.5], dtype=np.float32)}

            return actor

        agents = [
            _make_loaded_agent("zmq_sl", stateful=False, actor_fn=tracking_get_actor)
        ]
        thread = self._start_server_thread(zmq_context, address, agents)
        time.sleep(0.2)

        client = zmq_context.socket(zmq.REQ)
        client.connect(address)

        try:
            send_msg(client, {"type": MSG_INIT})
            resp = recv_msg(client)
            assert resp["type"] == MSG_INIT_RESPONSE

            obs_space = _simple_obs_space()
            act_space = _simple_act_space()
            send_msg(
                client,
                {
                    "type": MSG_SPACES,
                    "observation_spaces": {"0": obs_space},
                    "action_spaces": {"0": act_space},
                },
            )
            resp = recv_msg(client)
            assert resp["type"] == MSG_SPACES_RESPONSE

            for _ in range(2):
                obs = {"0": {"x": np.array([0.0, 0.0], dtype=np.float32)}}
                send_msg(client, {"type": MSG_STEP, "observations": obs})
                resp = recv_msg(client)
                assert resp["type"] == MSG_STEP_RESPONSE
                assert "0" in resp["actions"]

            send_msg(client, {"type": MSG_CLOSE})
            resp = recv_msg(client)
            assert resp["type"] == MSG_CLOSE_RESPONSE

            assert call_log == ["called", "called"]

        finally:
            client.close()
            thread.join(timeout=5)

    def test_zmq_mixed_agents(self, zmq_context):
        """ZMQ session with one stateless and one stateful agent."""
        zmq = pytest.importorskip("zmq")
        from pystk2_gymnasium.cli.race_protocol import recv_msg, send_msg

        address = "tcp://127.0.0.1:15568"

        stateless_log = []
        stateful_log = []

        def stateless_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                stateless_log.append("called")
                return {"acceleration": np.array([0.5], dtype=np.float32)}

            return actor

        def stateful_get_actor(module_dir, obs_space, act_space):
            def actor(state, obs):
                stateful_log.append(state["counter"])
                state["counter"] += 1
                return {"acceleration": np.array([0.8], dtype=np.float32)}

            return actor

        agents = [
            _make_loaded_agent("sl", stateful=False, actor_fn=stateless_get_actor),
            _make_loaded_agent("sf", stateful=True, actor_fn=stateful_get_actor),
        ]
        thread = self._start_server_thread(zmq_context, address, agents)
        time.sleep(0.2)

        client = zmq_context.socket(zmq.REQ)
        client.connect(address)

        try:
            send_msg(client, {"type": MSG_INIT})
            resp = recv_msg(client)
            assert resp["type"] == MSG_INIT_RESPONSE
            assert len(resp["agents"]) == 2

            obs_space = _simple_obs_space()
            act_space = _simple_act_space()
            send_msg(
                client,
                {
                    "type": MSG_SPACES,
                    "observation_spaces": {"0": obs_space, "1": obs_space},
                    "action_spaces": {"0": act_space, "1": act_space},
                },
            )
            resp = recv_msg(client)
            assert resp["type"] == MSG_SPACES_RESPONSE

            for _ in range(3):
                obs = {
                    "0": {"x": np.array([0.0, 0.0], dtype=np.float32)},
                    "1": {"x": np.array([0.1, 0.1], dtype=np.float32)},
                }
                send_msg(client, {"type": MSG_STEP, "observations": obs})
                resp = recv_msg(client)
                assert resp["type"] == MSG_STEP_RESPONSE
                assert "0" in resp["actions"]
                assert "1" in resp["actions"]

            send_msg(client, {"type": MSG_CLOSE})
            resp = recv_msg(client)
            assert resp["type"] == MSG_CLOSE_RESPONSE

            # Stateless agent: called 3 times with (None, obs)
            assert stateless_log == ["called"] * 3
            # Stateful agent: state counter increments 0, 1, 2
            assert stateful_log == [0, 1, 2]

        finally:
            client.close()
            thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_create_state_returning_none(self):
        """Agent with create_state that returns None — state is None."""
        la = _make_loaded_agent(
            "none_state",
            stateful=True,
            state_factory=lambda: None,
        )
        state = la.create_state()
        assert state is None

        # In the unified interface, create_state() is always called
        states = [la.create_state()]
        assert states[0] is None

    def test_runtime_reset_states_with_none_returning_create_state(self):
        """_AgentRuntime.reset_states handles create_state returning None."""
        la = _make_loaded_agent(
            "none_reset",
            stateful=True,
            state_factory=lambda: None,
        )
        runtime = _AgentRuntime([la], adapter_module=None)
        keys = ["0"]
        obs_spaces = {"0": _simple_obs_space()}
        act_spaces = {"0": _simple_act_space()}
        err = runtime.ensure_initialized(keys, obs_spaces, act_spaces)
        assert err is None

        states = runtime.reset_states()
        assert states["0"] is None

    def test_multiple_stateful_agents_independent_state(self):
        """Two stateful agents get independent state objects."""
        agents = [
            _make_loaded_agent("a", stateful=True),
            _make_loaded_agent("b", stateful=True),
        ]
        runtime = _AgentRuntime(agents, adapter_module=None)
        keys = ["0", "1"]
        obs_spaces = {k: _simple_obs_space() for k in keys}
        act_spaces = {k: _simple_act_space() for k in keys}
        runtime.ensure_initialized(keys, obs_spaces, act_spaces)

        states = runtime.reset_states()
        states["0"]["counter"] = 100
        assert states["1"]["counter"] == 0, "Agent states must be independent"

    def test_load_error_agent_create_state_returns_none(self):
        """Agent with load_error gets default create_state returning None."""
        la = LoadedAgent(
            env_name="supertuxkart/full-v0",
            player_name="broken",
            get_actor=None,
            module_dir=None,
            get_wrappers=None,
            source="broken",
            load_error="Failed to load",
        )
        runtime = _AgentRuntime([la], adapter_module=None)
        keys = ["0"]
        obs_spaces = {"0": _simple_obs_space()}
        act_spaces = {"0": _simple_act_space()}

        # Patch _build_wrapper_adapter since broken agents can't build wrappers
        with patch(
            "pystk2_gymnasium.cli.race_server._build_wrapper_adapter"
        ) as mock_bwa:
            mock_adapter = MagicMock()
            mock_adapter.observation_space = obs_spaces
            mock_adapter.action_space = act_spaces
            mock_bwa.return_value = mock_adapter

            err = runtime.ensure_initialized(keys, obs_spaces, act_spaces)
        assert err is None

        states = runtime.reset_states()
        assert states["0"] is None
