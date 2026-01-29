"""BBRL adapter for pystk2 race CLI.

Wraps a BBRL Agent into a simple obs -> action callable.

Usage:
    pystk2 race --adapter examples/bbrl_adapter.py my_bbrl_agent.zip

The agent's get_actor() should return a BBRL Agent instance.
This adapter handles workspace creation and action extraction.
"""

import torch


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


def wrap_actor(actor, obs_space, act_space):
    """Wrap a BBRL Agent into a callable(obs) -> action.

    :param actor: A BBRL Agent instance
    :param obs_space: The observation space
    :param act_space: The action space
    :returns: A callable that takes an observation dict and returns an action
    """
    from bbrl.agents.gymnasium import ParallelGymAgent
    from bbrl.workspace import Workspace

    # Put the agent in inference mode (disables dropout, etc.)
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
