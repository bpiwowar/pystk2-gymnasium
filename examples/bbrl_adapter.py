"""BBRL adapter for pystk2 race CLI.

Handles the full actor creation pipeline for BBRL agents:
loads weights, calls get_actor, wraps into an obs -> action callable.

Usage:
    pystk2 race --adapter examples/bbrl_adapter.py my_bbrl_agent.zip
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger("pystk2.adapter.bbrl")


def prepare_module_dir(path):
    """Ensure the agent directory is a proper Python package.

    Legacy BBRL projects may lack ``__init__.py`` while still using
    relative imports (e.g. ``from .wrappers import ...``).
    """
    init_path = Path(path) / "__init__.py"
    if not init_path.exists():
        logger.info("Creating missing __init__.py in %s", path)
        init_path.touch()


def get_action(workspace, t):
    """Extract action from a BBRL workspace at timestep t.

    Handles both tensor and nested dict action formats.
    Supports arbitrarily nested keys like ``action/group/steer``.
    """
    name = "action"
    if name in workspace.variables:
        return workspace.get(name, t)

    # Nested dict action: keys like "action/steer", "action/group/steer", etc.
    action = {}
    prefix = f"{name}/"
    len_prefix = len(prefix)
    for varname in workspace.variables:
        if not varname.startswith(prefix):
            continue
        keys = varname[len_prefix:].split("/")
        current = action
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = workspace.get(varname, t)
    return action


def dict_slice(k, obj):
    """Recursively index into a nested dict/tensor to remove batch dimension."""
    if isinstance(obj, dict):
        return {key: dict_slice(k, val) for key, val in obj.items()}
    return obj[k]


def create_actor(get_actor_fn, module_dir, obs_space, act_space):
    """Create a BBRL actor: load weights, build agent, wrap as callable.

    :param get_actor_fn: The agent's ``get_actor(state, obs_space, act_space)``
    :param module_dir: Path to the agent module directory
    :param obs_space: The observation space
    :param act_space: The action space
    :returns: A callable that takes an observation dict and returns an action
    """
    from bbrl.agents.gymnasium import ParallelGymAgent
    from bbrl.workspace import Workspace

    # Load weights from the agent's module directory
    pth_path = Path(module_dir) / "pystk_actor.pth"
    state = None
    if pth_path.is_file():
        state = torch.load(str(pth_path), map_location="cpu", weights_only=True)
        logger.info("Loaded weights from %s", pth_path)
    else:
        logger.warning("No pystk_actor.pth in %s", module_dir)

    # Create the BBRL agent
    actor = get_actor_fn(state, obs_space, act_space)

    # Put in inference mode (disables dropout, etc.)
    actor.train(False)

    workspace = Workspace()
    t = [0]  # mutable counter

    def call(obs):
        # Format observation into workspace
        # _format_frame already adds the batch dimension
        formatted = ParallelGymAgent._format_frame(obs)
        for key, value in formatted.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            workspace.set(f"env/{key}", t[0], value)

        # Run the BBRL agent (no gradient needed for inference)
        with torch.no_grad():
            actor(workspace, t=t[0])

        # Extract action and remove batch dimension
        action = get_action(workspace, t[0])
        t[0] += 1

        if isinstance(action, dict):
            return dict_slice(0, action)
        return action[0]

    return call
