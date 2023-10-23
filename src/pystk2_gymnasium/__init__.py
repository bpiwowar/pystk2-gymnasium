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

register(
    id="supertuxkart-flattened-v0",
    entry_point="pystk2_gymnasium.envs:SimpleSTKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec(
            "obs-flattener", "pystk2_gymnasium.wrappers:ObsFlattenerWrapper", {}
        ),
        WrapperSpec(
            "action-flattener", "pystk2_gymnasium.wrappers:ActionFlattenerWrapper", {}
        ),
    ),
)

# Discrete actions
register(
    id="supertuxkart-flattened-discrete-v0",
    entry_point="pystk2_gymnasium.envs:DiscreteActionSTKRaceEnv",
    max_episode_steps=1500,
    additional_wrappers=(
        WrapperSpec(
            "obs-flattener", "pystk2_gymnasium.wrappers:ObsFlattenerWrapper", {}
        ),
        WrapperSpec(
            "action-flattener", "pystk2_gymnasium.wrappers:ActionFlattenerWrapper", {}
        ),
    ),
)
