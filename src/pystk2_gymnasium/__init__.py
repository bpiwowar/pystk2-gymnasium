from gymnasium.envs.registration import register, WrapperSpec
from .definitions import ActionObservationWrapper, AgentSpec  # noqa: F401
from .wrappers import MonoAgentWrapperAdapter  # noqa: F401

# Version is setup automatically
__version__ = "0.0.0"
__version_tuple__ = (0, 0, 0)

register(
    id="supertuxkart/full-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(),
)

register(
    id="supertuxkart/multi-full-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceMultiEnv",
    max_episode_steps=1500,
    additional_wrappers=(),
)

register(
    id="supertuxkart/simple-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec(
            "constant-size",
            "pystk2_gymnasium.stk_wrappers:ConstantSizedObservations",
            {},
        ),
        WrapperSpec("polar", "pystk2_gymnasium.stk_wrappers:PolarObservations", {}),
    ),
)

# Flattens the spaces: join continuous actions, transform discrete into multidiscrete
# so that there are only two keys describing the environment
register(
    id="supertuxkart/flattened-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec(
            "constant-size",
            "pystk2_gymnasium.stk_wrappers:ConstantSizedObservations",
            {},
        ),
        WrapperSpec("polar", "pystk2_gymnasium.stk_wrappers:PolarObservations", {}),
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
    ),
)

# No discrete action (just steer/acceleration)
register(
    id="supertuxkart/flattened_continuous_actions-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec(
            "constant-size",
            "pystk2_gymnasium.stk_wrappers:ConstantSizedObservations",
            {},
        ),
        WrapperSpec("polar", "pystk2_gymnasium.stk_wrappers:PolarObservations", {}),
        WrapperSpec(
            "only-continous-actions",
            "pystk2_gymnasium.stk_wrappers:OnlyContinuousActionsWrapper",
            {},
        ),
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
    ),
)

# Multi-Discrete actions
register(
    id="supertuxkart/flattened_multidiscrete-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec(
            "constant-size",
            "pystk2_gymnasium.stk_wrappers:ConstantSizedObservations",
            {},
        ),
        WrapperSpec("polar", "pystk2_gymnasium.stk_wrappers:PolarObservations", {}),
        WrapperSpec(
            "discrete-actions",
            "pystk2_gymnasium.stk_wrappers:DiscreteActionsWrapper",
            {},
        ),
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
    ),
)

# Discrete actions
register(
    id="supertuxkart/flattened_discrete-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec(
            "constant-size",
            "pystk2_gymnasium.stk_wrappers:ConstantSizedObservations",
            {},
        ),
        WrapperSpec("polar", "pystk2_gymnasium.stk_wrappers:PolarObservations", {}),
        WrapperSpec(
            "discrete-actions",
            "pystk2_gymnasium.stk_wrappers:DiscreteActionsWrapper",
            {},
        ),
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
        WrapperSpec(
            "flatten-multi-discrete",
            "pystk2_gymnasium.wrappers:FlattenMultiDiscreteActions",
            {},
        ),
    ),
)
