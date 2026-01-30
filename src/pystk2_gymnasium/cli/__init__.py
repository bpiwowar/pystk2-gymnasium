"""CLI entry point for pystk2 commands."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="pystk2",
        description="SuperTuxKart 2 CLI tools",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- race subcommand ---
    race_parser = subparsers.add_parser(
        "race", help="Run a race with one or more agents"
    )
    race_parser.add_argument(
        "agents",
        nargs="+",
        help=(
            "Agent sources: path to a zip file, a directory containing "
            "pystk_actor.py, or a Python module name"
        ),
    )
    race_parser.add_argument(
        "--num-karts",
        type=int,
        default=3,
        help="Total number of karts in the race (default: 3)",
    )
    race_parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Maximum number of path nodes ahead (default: unlimited)",
    )
    race_parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Track name (default: random)",
    )
    race_parser.add_argument(
        "--laps",
        type=int,
        default=1,
        help="Number of laps (default: 1)",
    )
    race_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON race results",
    )
    race_parser.add_argument(
        "--error-handling",
        choices=["raise", "catch"],
        default="raise",
        help=(
            "How to handle agent errors: 'raise' to propagate, "
            "'catch' to use random actions (default: raise)"
        ),
    )
    race_parser.add_argument(
        "--action-timeout",
        type=float,
        default=None,
        help="Per-action timeout in seconds (Unix only, default: no timeout)",
    )
    race_parser.add_argument(
        "--hide",
        action="store_true",
        help="Run without graphics (headless mode)",
    )
    race_parser.add_argument(
        "--web",
        action="store_true",
        help="Enable web-based visualization dashboard (requires dash/plotly)",
    )
    race_parser.add_argument(
        "--web-port",
        type=int,
        default=8050,
        help="Port for the web dashboard (default: 8050)",
    )
    race_parser.add_argument(
        "--record",
        type=str,
        metavar="FILE",
        default=None,
        help="Save race video to FILE (e.g. race.mp4). All cameras tiled into one video.",
    )
    race_parser.add_argument(
        "--cameras",
        type=int,
        default=None,
        help="Number of cameras (max 8, default: min(num_karts, 8) when recording)",
    )
    race_parser.add_argument(
        "--screen-width",
        type=int,
        default=None,
        help="Camera screen width in pixels (default: pystk2 HD preset)",
    )
    race_parser.add_argument(
        "--screen-height",
        type=int,
        default=None,
        help="Camera screen height in pixels (default: pystk2 HD preset)",
    )
    race_parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frame rate (default: 20)",
    )
    race_parser.add_argument(
        "--adapter",
        type=str,
        metavar="PATH",
        default=None,
        help="Python file providing create_actor(get_actor, module_dir, obs_space, act_space)",
    )
    race_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps before stopping the race (default: no limit)",
    )

    # --- race-server subcommand ---
    server_parser = subparsers.add_parser(
        "race-server",
        help="Start a race server that serves agent actions over ZMQ",
    )
    server_parser.add_argument(
        "agents",
        nargs="+",
        help=(
            "Agent sources: path to a zip file, a directory containing "
            "pystk_actor.py, or a Python module name"
        ),
    )
    server_parser.add_argument(
        "--address",
        type=str,
        default="tcp://*:5555",
        help="ZMQ bind address (default: tcp://*:5555)",
    )
    server_parser.add_argument(
        "--adapter",
        type=str,
        metavar="PATH",
        default=None,
        help="Python file providing create_actor(get_actor, module_dir, obs_space, act_space)",
    )
    server_parser.add_argument(
        "--action-timeout",
        type=float,
        default=None,
        help="Per-action timeout in seconds (Unix only, default: no timeout)",
    )
    server_parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Number of worker threads for concurrent client sessions "
            "(default: half the CPU cores)"
        ),
    )

    # --- race-client subcommand ---
    client_parser = subparsers.add_parser(
        "race-client",
        help="Run a race using remote agent servers over ZMQ",
    )
    client_parser.add_argument(
        "--server",
        type=str,
        action="append",
        required=True,
        help=(
            "Server address to connect to (e.g. tcp://localhost:5555). "
            "Repeat for multiple servers."
        ),
    )
    client_parser.add_argument(
        "--num-karts",
        type=int,
        default=3,
        help="Total number of karts in the race (default: 3)",
    )
    client_parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Maximum number of path nodes ahead (default: unlimited)",
    )
    client_parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Track name (default: random)",
    )
    client_parser.add_argument(
        "--laps",
        type=int,
        default=1,
        help="Number of laps (default: 1)",
    )
    client_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON race results",
    )
    client_parser.add_argument(
        "--error-handling",
        choices=["raise", "catch"],
        default="raise",
        help=(
            "How to handle agent errors: 'raise' to propagate, "
            "'catch' to use random actions (default: raise)"
        ),
    )
    client_parser.add_argument(
        "--hide",
        action="store_true",
        help="Run without graphics (headless mode)",
    )
    client_parser.add_argument(
        "--web",
        action="store_true",
        help="Enable web-based visualization dashboard (requires dash/plotly)",
    )
    client_parser.add_argument(
        "--web-port",
        type=int,
        default=8050,
        help="Port for the web dashboard (default: 8050)",
    )
    client_parser.add_argument(
        "--record",
        type=str,
        metavar="FILE",
        default=None,
        help="Save race video to FILE (e.g. race.mp4). All cameras tiled into one video.",
    )
    client_parser.add_argument(
        "--cameras",
        type=int,
        default=None,
        help="Number of cameras (max 8, default: min(num_karts, 8) when recording)",
    )
    client_parser.add_argument(
        "--screen-width",
        type=int,
        default=None,
        help="Camera screen width in pixels (default: pystk2 HD preset)",
    )
    client_parser.add_argument(
        "--screen-height",
        type=int,
        default=None,
        help="Camera screen height in pixels (default: pystk2 HD preset)",
    )
    client_parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frame rate (default: 20)",
    )
    client_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps before stopping the race (default: no limit)",
    )
    client_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="ZMQ recv timeout per request in seconds (default: 60)",
    )
    client_parser.add_argument(
        "--max-steps-after-first",
        type=int,
        default=None,
        help=(
            "Maximum steps to continue after the first kart finishes "
            "(default: no limit)"
        ),
    )
    client_parser.add_argument(
        "--karts-finished",
        type=int,
        default=None,
        help="Stop the race after this many karts have finished (default: all)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "race":
        # Lazy import to avoid pulling in heavy dependencies for --help
        from pystk2_gymnasium.cli.race import run_race

        run_race(args)

    elif args.command == "race-server":
        from pystk2_gymnasium.cli.race_server import run_race_server

        run_race_server(args)

    elif args.command == "race-client":
        from pystk2_gymnasium.cli.race_client import run_race_client

        run_race_client(args)
