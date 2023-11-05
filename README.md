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

- `supertuxkart-v0` is the main environment containing complete observations. The observation and action spaces are both dictionaries with continuous or discrete variables (see below). The exact structure can be found using `env.observation_space` and `env.action_space`. The following options can be used to modify the environment:
    - `render_mode` can be None or `human`
    - `track` defines the SuperTuxKart track to use (None for random). The full list can be found in `STKRaceEnv.TRACKS` after initialization with `initialize.initialize(with_graphics: bool)` has been called.
    - `num_kart` defines the number of karts on the track (3 by default)
    - `rank_start` defines the starting position (None for random, which is the default)
    - `use_ai` flag (False by default) to ignore actions (when calling `step`, and use a SuperTuxKart bot)
    - `max_paths` the maximum number of the (nearest) paths (a track is made of paths) to consider in the observation state
    - `laps` is the number of laps (1 by default)
    - `difficulty` is the difficulty of the AI bots (lowest 0 to highest 2, default to 2)
- `supertuxkart-simple-v0` is a simplified environment with a fixed number of observations for paths (controlled by `state_paths`, default 5), items (`state_items`, default 5), karts (`state_karts`, default 5)
- `supertuxkart-flattened-v0` has observation and action spaces simplified at the maximum (only `discrete` and `continuous` keys)
- `supertuxkart-flattened-continuous-actions-v0` removes discrete actions (default to 0) so this is steer/acceleration only in the continuous domain
- `supertuxkart-flattened-multidiscrete-v0` is like the previous one, but with fully multi-discrete actions. `acceleration_steps` and `steer_steps` (default to 5) control the number of discrete values for acceleration and steering respectively.
- `supertuxkart-flattened-discrete-v0` is like the previous one, but with fully discretized actions

The reward $r_t$ at time $t$ is given by

$$ r_{t} =  \frac{1}{10}(d_{t} - d_{t-1}) + (1 - \frac{\mathrm{pos}_t}{K}) \times (3 + 7 f_t) - 0.1 + 10 * f_t $$

where $d_t$ is the
overall track distance at time $t$, $\mathrm{pos}_t$ the position among the $K$ karts at time $t$, and $f_t$ is $1$ when the kart finishes the race.

## Action and observation space

All the 3D vectors are within the kart referential (`z` front, `x` left, `y` up):

- `distance_down_track`: The distance from the start
- `energy`: remaining collected energy
- `front`: front of the kart (3D vector)
- `items_position`: position of the items (3D vectors)
- `attachment`: the item attached to the kart (bonus box, banana, nitro/big, nitro/small, bubble gum, easter egg)
- `attachment_time_left`: how much time the attachment will be kept
- `items_type`: type of the item
- `jumping`: is the kart jumping
- `karts_position`: position of other karts, beginning with the ones in front
- `max_steer_angle` the max angle of the steering (given the current speed)
- `paths_distance`: the distance of the paths
- `paths_start`, `paths_end`, `paths_width`: 3D vector to the paths start and end, with their widths (sccalar)
- `paths_start`: 3D vectors to the the path s
- `powerup`
- `shield_time`
- `skeed_factor`
- `velocity`: velocity vector

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
