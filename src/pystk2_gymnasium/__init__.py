from gymnasium.envs.registration import register, WrapperSpec

register(
    id="supertuxkart-v0",
    entry_point="pystk2_gymnasium.envs:STKRaceEnv",
    max_episode_steps=1500,
)

register(
    id="supertuxkart-simple-v0",
    entry_point="pystk2_gymnasium.envs:SimpleSTKRaceEnv",
    max_episode_steps=1500,
)

# Flattens the spaces: join continuous actions, transform discrete into multidiscrete
# so that there are only two keys describing the environment
register(
    id="supertuxkart-flattened-v0",
    entry_point="pystk2_gymnasium.envs:SimpleSTKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
    ),
)

# No discrete action (just steer/acceleration)
register(
    id="supertuxkart-flattened-continuous-actions-v0",
    entry_point="pystk2_gymnasium.envs:OnlyContinuousActionSTKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
    ),
)

# Multi-Discrete actions
register(
    id="supertuxkart-flattened-multidiscrete-v0",
    entry_point="pystk2_gymnasium.envs:DiscreteActionSTKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
    ),
)

# Discrete actions
register(
    id="supertuxkart-flattened-discrete-v0",
    entry_point="pystk2_gymnasium.envs:DiscreteActionSTKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenerWrapper", {}),
        WrapperSpec("obs-flattener", "pystk2_gymnasium.wrappers:FlattenMultiDiscreteActions", {}),
    ),
)
