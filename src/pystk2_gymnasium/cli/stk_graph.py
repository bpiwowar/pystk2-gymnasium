"""Web-based visualization dashboard for STK races using Dash/Plotly."""

import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("pystk2.cli.web")


class RaceController:
    """Thread-safe controller for the race loop.

    The race loop calls :meth:`wait_for_step` before each iteration.
    The web UI sets the running/step/stop state via button callbacks.

    Default state is **paused** so the user sees the initial state.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = False
        self._step_requested = False
        self._stop_requested = False

    # -- called from race loop (main thread) --

    def wait_for_step(self) -> bool:
        """Block until a step is allowed. Returns False if stop was requested."""
        with self._condition:
            while True:
                if self._stop_requested:
                    return False
                if self._running:
                    return True
                if self._step_requested:
                    self._step_requested = False
                    return True
                self._condition.wait()

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stop_requested

    # -- called from Dash callbacks (server thread) --

    def request_run(self):
        with self._condition:
            self._running = True
            self._condition.notify_all()

    def request_pause(self):
        with self._condition:
            self._running = False

    def request_step(self):
        with self._condition:
            self._step_requested = True
            self._condition.notify_all()

    def request_stop(self):
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running


def _serialize_obs(obs):
    """Convert observations to a JSON-friendly nested structure.

    Handles numpy arrays, tuples of arrays, dicts, scalars.
    """
    if isinstance(obs, dict):
        return {k: _serialize_obs(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        if obs.ndim == 0:
            return float(obs)
        if obs.size <= 12:
            return obs.tolist()
        # Truncate large arrays
        return {"shape": list(obs.shape), "values": obs.flat[:8].tolist(), "...": True}
    if isinstance(obs, (tuple, list)):
        items = [_serialize_obs(v) for v in obs]
        if len(items) > 20:
            return items[:20] + ["..."]
        return items
    if isinstance(obs, (np.integer, np.bool_)):
        return int(obs)
    if isinstance(obs, np.floating):
        return round(float(obs), 4)
    if isinstance(obs, (int, float, bool, str)):
        return obs
    return str(obs)


def _render_obs_tree(value, label: str = "") -> List:
    """Render a serialized observation as collapsible HTML <details> nodes."""
    from dash import html

    if isinstance(value, dict):
        # Dict -> collapsible node with children
        children = []
        for k, v in sorted(value.items()):
            children.extend(_render_obs_tree(v, label=str(k)))
        summary_text = f"{label}" if label else "obs"
        return [
            html.Details(
                [html.Summary(summary_text, style={"cursor": "pointer"})] + children,
                style={"marginLeft": "12px"},
            )
        ]
    if (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], (dict, list))
    ):
        # List of compound items -> collapsible with indexed children
        children = []
        for i, item in enumerate(value):
            if item == "...":
                children.append(
                    html.Div("...", style={"marginLeft": "12px", "color": "#888"})
                )
            else:
                children.extend(_render_obs_tree(item, label=f"[{i}]"))
        summary_text = (
            f"{label} ({len(value)} items)" if label else f"({len(value)} items)"
        )
        return [
            html.Details(
                [html.Summary(summary_text, style={"cursor": "pointer"})] + children,
                style={"marginLeft": "12px"},
            )
        ]
    # Leaf value -> single line
    formatted = _format_leaf(value)
    if label:
        return [
            html.Div(
                [
                    html.Span(f"{label}: ", style={"color": "#555"}),
                    html.Span(formatted),
                ],
                style={"marginLeft": "12px", "whiteSpace": "nowrap"},
            )
        ]
    return [html.Div(formatted, style={"marginLeft": "12px"})]


def _format_leaf(value) -> str:
    """Format a leaf value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        formatted = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in value]
        return "[" + ", ".join(formatted) + "]"
    return str(value)


class WebDashboard:
    """Dash/Plotly web dashboard running in a background daemon thread.

    The race loop pushes state via ``update()``. A Dash interval callback
    reads the latest state under a lock and refreshes the page.

    Exposes a :class:`RaceController` for step/run/pause/stop from the UI.
    """

    def __init__(
        self,
        port: int = 8050,
        num_controlled: int = 0,
        agent_names: Optional[List[str]] = None,
    ):
        self.port = port
        self.num_controlled = num_controlled
        self.agent_names = agent_names or []
        self.controller = RaceController()
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._track_data: Optional[Dict[str, Any]] = None
        self._app = None
        self._thread = None

    # ------------------------------------------------------------------
    # Public API (called from main thread)
    # ------------------------------------------------------------------

    def update(self, env, obs, info, total_rewards, step_count):
        """Push latest race state (called from the race loop)."""
        unwrapped = env.unwrapped

        # Cache track geometry once
        if self._track_data is None and hasattr(unwrapped, "track"):
            self._track_data = self._extract_track(unwrapped.track)

        # Controlled kart indices (from STKRaceMultiEnv)
        kart_indices = getattr(unwrapped, "kart_indices", [])

        data = {
            "step": step_count,
            "total_rewards": dict(total_rewards),
            "karts": self._extract_karts(unwrapped),
            "items": self._extract_items(unwrapped),
            "agent_infos": info.get("infos", {}),
            "observations": _serialize_obs(obs),
            "controlled_kart_indices": list(kart_indices),
        }

        with self._lock:
            self._data = data

    def start(self):
        """Build the Dash app and start in a daemon thread."""
        self._app = self._build_app()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_track(track):
        """Extract track geometry (called once, cached)."""
        xs, ys, zs = [], [], []
        for node in track.path_nodes:
            start, end = node
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            # Map STK coords to plot: X->X, Z->Y (horizontal), Y->Z (up)
            xs.append(mid_x)
            ys.append(mid_z)
            zs.append(mid_y)
        return {"x": xs, "y": ys, "z": zs}

    @staticmethod
    def _extract_karts(unwrapped):
        """Extract current kart positions and names."""
        if not hasattr(unwrapped, "world") or unwrapped.world is None:
            return []
        karts = []
        for kart in unwrapped.world.karts:
            loc = kart.location
            karts.append(
                {
                    "name": kart.name,
                    "x": loc[0],
                    "y": loc[2],  # STK Z -> plot Y
                    "z": loc[1],  # STK Y -> plot Z
                    "position": kart.position,
                    "finished": kart.has_finished_race,
                    "distance": max(0, kart.overall_distance),
                    "energy": kart.energy,
                    "velocity": float(
                        (
                            kart.velocity_lc[0] ** 2
                            + kart.velocity_lc[1] ** 2
                            + kart.velocity_lc[2] ** 2
                        )
                        ** 0.5
                    ),
                }
            )
        return karts

    @staticmethod
    def _extract_items(unwrapped):
        """Extract item positions."""
        if not hasattr(unwrapped, "world") or unwrapped.world is None:
            return []
        items = []
        for item in unwrapped.world.items:
            loc = item.location
            items.append(
                {
                    "type": str(item.type),
                    "x": loc[0],
                    "y": loc[2],
                    "z": loc[1],
                }
            )
        return items

    # ------------------------------------------------------------------
    # Dash app
    # ------------------------------------------------------------------

    def _build_app(self):
        from dash import Dash, html, dcc, ctx, no_update
        from dash.dependencies import Input, Output

        app = Dash(
            __name__,
            update_title=None,
            suppress_callback_exceptions=True,
        )
        app.title = "STK Race Dashboard"

        button_style = {
            "padding": "8px 20px",
            "fontSize": "14px",
            "cursor": "pointer",
            "border": "1px solid #555",
            "borderRadius": "4px",
            "marginRight": "8px",
        }

        app.layout = html.Div(
            [
                html.H1("SuperTuxKart Race Dashboard"),
                # Control bar
                html.Div(
                    [
                        html.Button(
                            "Step",
                            id="btn-step",
                            n_clicks=0,
                            style={**button_style, "background": "#e0e0e0"},
                        ),
                        html.Button(
                            "Run",
                            id="btn-run",
                            n_clicks=0,
                            style={**button_style, "background": "#b5e8b5"},
                        ),
                        html.Button(
                            "Pause",
                            id="btn-pause",
                            n_clicks=0,
                            style={**button_style, "background": "#f0d080"},
                        ),
                        html.Button(
                            "Stop",
                            id="btn-stop",
                            n_clicks=0,
                            style={**button_style, "background": "#f0a0a0"},
                        ),
                        html.Span(
                            "PAUSED",
                            id="status-label",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "marginLeft": "12px",
                                "verticalAlign": "middle",
                            },
                        ),
                    ],
                    style={"marginBottom": "12px"},
                ),
                dcc.Interval(id="interval", interval=500, n_intervals=0),
                dcc.Graph(
                    id="track-3d",
                    style={"height": "600px", "width": "100%"},
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["toImage"],
                    },
                ),
                html.Div(
                    id="info-panel",
                    style={
                        "padding": "10px",
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "10px",
                    },
                ),
            ],
            style={"padding": "20px", "fontFamily": "monospace"},
        )

        # --- Control button callbacks ---
        @app.callback(
            Output("status-label", "children"),
            [
                Input("btn-step", "n_clicks"),
                Input("btn-run", "n_clicks"),
                Input("btn-pause", "n_clicks"),
                Input("btn-stop", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def _handle_controls(_step, _run, _pause, _stop):
            triggered = ctx.triggered_id
            if triggered == "btn-step":
                self.controller.request_step()
                return "STEP"
            elif triggered == "btn-run":
                self.controller.request_run()
                return "RUNNING"
            elif triggered == "btn-pause":
                self.controller.request_pause()
                return "PAUSED"
            elif triggered == "btn-stop":
                self.controller.request_stop()
                return "STOPPED"
            return no_update

        # --- Data refresh callback ---
        @app.callback(
            [Output("track-3d", "figure"), Output("info-panel", "children")],
            [Input("interval", "n_intervals")],
        )
        def _refresh(_n):
            return self._make_figure(), self._make_info_panel()

        return app

    def _make_figure(self):
        import plotly.graph_objects as go

        fig = go.Figure()

        # Track path
        td = self._track_data
        if td:
            fig.add_trace(
                go.Scatter3d(
                    x=td["x"],
                    y=td["y"],
                    z=td["z"],
                    mode="lines",
                    line=dict(color="gray", width=3),
                    name="Track",
                    hoverinfo="skip",
                )
            )

        with self._lock:
            data = dict(self._data)

        karts = data.get("karts", [])
        items = data.get("items", [])
        controlled = set(data.get("controlled_kart_indices", []))

        # Build display names: use agent name for controlled karts
        kart_to_agent = {}
        for agent_idx, kart_idx in enumerate(data.get("controlled_kart_indices", [])):
            kart_to_agent[kart_idx] = agent_idx

        kart_labels = []
        for ix, k in enumerate(karts):
            agent_idx = kart_to_agent.get(ix)
            if agent_idx is not None and agent_idx < len(self.agent_names):
                kart_labels.append(self.agent_names[agent_idx])
            else:
                kart_labels.append(k["name"])

        # Kart markers
        if karts:
            colors = ["blue" if ix in controlled else "red" for ix in range(len(karts))]
            fig.add_trace(
                go.Scatter3d(
                    x=[k["x"] for k in karts],
                    y=[k["y"] for k in karts],
                    z=[k["z"] for k in karts],
                    mode="markers+text",
                    marker=dict(size=8, color=colors),
                    text=kart_labels,
                    textposition="top center",
                    name="Karts",
                )
            )

        # Item markers
        if items:
            fig.add_trace(
                go.Scatter3d(
                    x=[i["x"] for i in items],
                    y=[i["y"] for i in items],
                    z=[i["z"] for i in items],
                    mode="markers",
                    marker=dict(size=4, color="green", symbol="diamond"),
                    text=[i["type"] for i in items],
                    name="Items",
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Z (forward)",
                zaxis_title="Y (up)",
                aspectmode="data",
                # Preserve 3D camera across figure updates
                uirevision="keep",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            # Preserve all UI state across figure updates
            uirevision="keep",
        )
        return fig

    def _make_info_panel(self):
        from dash import html

        with self._lock:
            data = dict(self._data)

        if not data:
            return html.P("Waiting for race data...")

        children = [html.H3(f"Step {data.get('step', '?')}")]

        karts = data.get("karts", [])
        total_rewards = data.get("total_rewards", {})
        observations = data.get("observations", {})
        # controlled_kart_indices: agent_idx -> world_kart_idx
        # Build reverse map: world_kart_idx -> agent_idx
        kart_to_agent = {}
        for agent_idx, kart_idx in enumerate(data.get("controlled_kart_indices", [])):
            kart_to_agent[kart_idx] = agent_idx

        for ix, kart in enumerate(karts):
            agent_idx = kart_to_agent.get(ix)
            is_controlled = agent_idx is not None
            agent_key = str(agent_idx) if is_controlled else None
            reward = total_rewards.get(agent_key) if agent_key else None

            if is_controlled and agent_idx < len(self.agent_names):
                display_name = self.agent_names[agent_idx]
            else:
                display_name = kart["name"]
            tag = " [agent]" if is_controlled else " [AI]"
            card_children = [
                html.Strong(f"{display_name} (P{kart['position']}){tag}"),
                html.Br(),
                html.Span(f"Distance: {kart['distance']:.1f}"),
                html.Br(),
                html.Span(f"Energy: {kart['energy']:.1f}"),
                html.Br(),
                html.Span(f"Speed: {kart['velocity']:.1f}"),
            ]
            if reward is not None:
                card_children.extend(
                    [html.Br(), html.Span(f"Total reward: {reward:.2f}")]
                )
            if kart["finished"]:
                card_children.extend([html.Br(), html.B("FINISHED")])

            # Observation tree for this agent
            agent_obs = observations.get(agent_key) if agent_key else None
            if agent_obs is not None:
                card_children.append(html.Hr(style={"margin": "6px 0"}))
                card_children.extend(_render_obs_tree(agent_obs, label="observation"))

            if is_controlled:
                card_style = {
                    "border": "2px solid #2a7fff",
                    "background": "#e8f0fe",
                    "padding": "8px",
                    "borderRadius": "4px",
                    "minWidth": "220px",
                    "flex": "1",
                }
            else:
                card_style = {
                    "border": "1px solid #ccc",
                    "background": "#f8f8f8",
                    "padding": "8px",
                    "borderRadius": "4px",
                    "minWidth": "220px",
                    "flex": "1",
                }

            children.append(html.Div(card_children, style=card_style))

        return children

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    def _run_server(self):
        # Suppress Dash/Flask startup banner
        flask_log = logging.getLogger("werkzeug")
        flask_log.setLevel(logging.WARNING)

        self._app.run(
            host="0.0.0.0",
            port=self.port,
            debug=False,
            dev_tools_hot_reload=False,
        )
