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

    Handles both tensor and dict (nested) action formats.
    """
    keys = [k for k in workspace.keys() if k.startswith("action")]
    if len(keys) == 1 and keys[0] == "action":
        return workspace.get("action", t)

    # Nested dict action: keys like "action/steer", "action/acceleration", etc.
    action = {}
    prefix = "action/"
    for k in keys:
        if k.startswith(prefix):
            name = k[len(prefix) :]
            action[name] = workspace.get(k, t)
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
    from bbrl.workspace import Workspace

    try:
        from bbrl_gymnasium.agents.gymnasium import ParallelGymAgent
    except ImportError:
        from bbrl_gymnasium.agents.gymagent import ParallelGymAgent

    workspace = Workspace()
    t = [0]  # mutable counter

    def call(obs):
        # Format observation into workspace
        formatted = ParallelGymAgent._format_frame(obs)
        for key, value in formatted.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            # Add batch dimension
            if value.dim() == 0:
                value = value.unsqueeze(0)
            else:
                value = value.unsqueeze(0)
            workspace.set(f"env/{key}", t[0], value)

        # Run the BBRL agent
        actor(workspace, t=t[0])

        # Extract action
        action = get_action(workspace, t[0])
        t[0] += 1

        # Remove batch dimension
        return dict_slice(0, action)

    return call
