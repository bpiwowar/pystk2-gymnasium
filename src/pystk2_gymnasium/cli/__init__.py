"""CLI entry point for pystk2 commands."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="pystk2",
        description="SuperTuxKart 2 CLI tools",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

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

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "race":
        # Lazy import to avoid pulling in heavy dependencies for --help
        from pystk2_gymnasium.cli.race import run_race

        run_race(args)
