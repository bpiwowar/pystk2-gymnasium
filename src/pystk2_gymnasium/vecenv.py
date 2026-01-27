"""Vectorized environment support for pystk2-gymnasium.

This module provides utilities to create vectorized environments that run
multiple STK races in parallel using Gymnasium's AsyncVectorEnv.
"""

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv


def _has_dynamic_spaces(space: gym.Space) -> bool:
    """Check if a space contains any dynamic (variable-length) subspaces.

    Dynamic spaces like Sequence cannot use shared memory in AsyncVectorEnv.
    """
    if isinstance(space, (spaces.Sequence, spaces.Graph)):
        return True
    elif isinstance(space, spaces.Dict):
        return any(_has_dynamic_spaces(subspace) for subspace in space.values())
    elif isinstance(space, spaces.Tuple):
        return any(_has_dynamic_spaces(subspace) for subspace in space)
    return False


def make_stkrace_vec(
    num_envs: int,
    env_id: str = "supertuxkart/simple-v0",
    env_kwargs: Optional[Dict[str, Any]] = None,
    context: Optional[str] = "spawn",
    shared_memory: Optional[bool] = None,
    **vec_kwargs,
) -> AsyncVectorEnv:
    """Create a vectorized STK environment with parallel races.

    Each environment runs in a separate process, allowing true parallel
    execution of multiple races. This is necessary because STK only allows
    one race instance per process.

    Args:
        num_envs: Number of parallel environments to create.
        env_id: The registered environment ID to use.
            Defaults to "supertuxkart/simple-v0".
        env_kwargs: Additional keyword arguments passed to the environment.
            Note: `use_subprocess` is automatically set to False.
        context: Multiprocessing context. Defaults to "spawn" for clean
            process isolation. Can also be "fork" (Unix only) or "forkserver".
        shared_memory: Whether to use shared memory for observations.
            If None (default), auto-detects based on observation space:
            enabled for fixed-size observations (e.g., with ConstantSizedObservations
            wrapper), disabled for variable-length observations (Sequence spaces).
        **vec_kwargs: Additional arguments passed to AsyncVectorEnv.

    Returns:
        An AsyncVectorEnv instance with num_envs parallel STK environments.

    Example:
        >>> import pystk2_gymnasium
        >>> vec_env = pystk2_gymnasium.make_stkrace_vec(
        ...     num_envs=4,
        ...     env_id="supertuxkart/simple-v0",
        ...     env_kwargs={"track": "lighthouse", "num_kart": 3}
        ... )
        >>> observations, infos = vec_env.reset()
        >>> # observations shape: (4, ...) for 4 parallel envs
        >>> vec_env.close()
    """
    if env_kwargs is None:
        env_kwargs = {}

    # Force use_subprocess=False since each AsyncVectorEnv worker is already
    # a separate process
    env_kwargs = {**env_kwargs, "use_subprocess": False}

    # Auto-detect shared_memory based on observation space
    if shared_memory is None:
        # Check if observation space has dynamic (variable-length) components
        # by inspecting the registered environment's observation space
        spec = gym.envs.registry.get(env_id)
        if spec is not None:
            # Create a temporary env to check observation space
            # Use the spec to get observation space without full initialization
            temp_env = gym.make(env_id, **env_kwargs)
            try:
                has_dynamic = _has_dynamic_spaces(temp_env.observation_space)
                shared_memory = not has_dynamic
            finally:
                temp_env.close()
        else:
            # Unknown env, default to False for safety
            shared_memory = False

    def make_env(env_id: str, env_kwargs: Dict[str, Any]) -> Callable[[], gym.Env]:
        """Create an environment factory function."""

        def _make() -> gym.Env:
            # Import pystk2_gymnasium to register environments in spawned process
            import pystk2_gymnasium  # noqa: F401

            return gym.make(env_id, **env_kwargs)

        return _make

    env_fns = [make_env(env_id, env_kwargs) for _ in range(num_envs)]

    return AsyncVectorEnv(
        env_fns, context=context, shared_memory=shared_memory, **vec_kwargs
    )
