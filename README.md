# PySuperTuxKart gymnasium wrapper

*warning*: pystk2-gymnasium is in alpha stage - the environments might change abruptly!

## Install

The PySuperKart2 gymnasium wrapper is a Python package, so installing is fairly easy

`pip install pystk2-gymnasium`

Note that during the first run, SuperTuxKart assets are downloaded in the cache directory.

## Environments

*Warning* only one SuperTuxKart environment can be created for now. Moreover, no graphics information 
is available for now.

After importing `pystk2_gymnasium`, the following environments are available:

- `supertuxkart-v0` is the main environment containing complete observations, with the following options:
    - `render_mode` can be None or `human`
    - `track` defines the SuperTuxKart track to use (None for random). The full list can be found in `STKRaceEnv.TRACKS` after initialization with `initialize.initialize(with_graphics: bool)` has been called.
    - `num_kart` defines the number of karts on the track (3 by default)
    - `rank_start` defines the starting position (None for random, which is the default)
    - `use_ai` flag (False by default) to ignore actions and use a SuperTuxKart bot
    - `max_paths` the maximum number of the (nearest) paths (a track is made of paths) to consider in the observation state
    - `laps` is the number of laps (1 by default)
    - `difficulty` is the difficulty of the other bots (0 to 2, default to 2)
- `supertuxkart-simple-v0` is a simplified environment with fixed number observations for paths (controlled by `state_paths`, default 5), items (`state_items`, default 5), karts (`state_karts`, default 5)
- `supertuxkart-flattened-v0` has observation and action spaces simplified at the maximum (only `discrete` and `continuous` keys)
- `supertuxkart-flattened-discrete-v0` is like the previous one, but with fully discretized actions

## Example

```py3
import gymnasium as gym
import pystk2_gymnasium

# Use a a flattened version of the observation and action spaces
# In both case, this corresponds to a dictionary with two keys:
# - `continuous` is a vector corresponding to the continuous observations
# - `discrete` is a vector (of integers) corresponding to discrete observations
env = gym.make("supertuxkart-flattened-v0", render_mode="human", use_ai=False)

ix = 0
done = False
state, *_ = env.reset()

while not done:
    ix += 1
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    done = truncated or terminated
```
