# PySuperTuxKart gymnasium wrapper

[![PyPI
version](https://badge.fury.io/py/pystk2-gymnasium.svg)](https://badge.fury.io/py/pystk2-gymnasium)

Read the [Changelog](./CHANGELOG.md)

## Install

The PySuperKart2 gymnasium wrapper is a Python package, so installing is fairly
easy

`pip install pystk2-gymnasium`

Note that during the first run, SuperTuxKart assets are downloaded in the cache
directory.

## AgentSpec

Each controlled kart is parametrized by `pystk2_gymnasium.AgentSpec`:

- `name` defines name of the player (displayed on top of the kart)
- `rank_start` defines the starting position (None for random, which is the
  default)
- `use_ai` flag (False by default) to ignore actions (when calling `step`,  a
  SuperTuxKart bot is used instead of using the action)
- `camera_mode` can be set to `AUTO` (camera on for non STK bots), `ON` (camera
  on) or `OFF` (no camera).


## Current limitations

-  no graphics information is available (i.e. pixmap)


## Environments

After importing `pystk2_gymnasium`, the following environments are available:

- `supertuxkart/full-v0` is the main environment containing complete
  observations. The observation and action spaces are both dictionaries with
  continuous or discrete variables (see below). The exact structure can be found
  using `env.observation_space` and `env.action_space`. The following options
  can be used to modify the environment:
    - `agent` is an `AgentSpec (see above)`
    - `render_mode` can be None or `human`
    - `track` defines the SuperTuxKart track to use (None for random). The full
      list can be found in `STKRaceEnv.TRACKS` after initialization with
      `initialize.initialize(with_graphics: bool)` has been called.
    - `num_kart` defines the number of karts on the track (3 by default)
    - `max_paths` the maximum number of the (nearest) paths (a track is made of
      paths) to consider in the observation state
    - `laps` is the number of laps (1 by default)
    - `difficulty` is the difficulty of the AI bots (lowest 0 to highest 2,
      default to 2)

Some environments are created using wrappers (see below for wrapper
documentation),
- `supertuxkart/simple-v0` (wrappers: `ConstantSizedObservations`) is a
  simplified environment with a fixed number of observations for paths
  (controlled by `state_paths`, default 5), items (`state_items`, default 5),
  karts (`state_karts`, default 5)
- `supertuxkart/flattened-v0` (wrappers: `ConstantSizedObservations`,
  `PolarObservations`, `FlattenerWrapper`) has observation and action spaces
  simplified at the maximum (only `discrete` and `continuous` keys)
- `supertuxkart/flattened_continuous_actions-v0` (wrappers: `ConstantSizedObservations`, `PolarObservations`, `OnlyContinuousActionsWrapper`, `FlattenerWrapper`) removes discrete actions
  (default to 0) so this is steer/acceleration only in the continuous domain
- `supertuxkart/flattened_multidiscrete-v0` (wrappers: `ConstantSizedObservations`, `PolarObservations`, `DiscreteActionsWrapper`, `FlattenerWrapper`) is like the previous one, but with
  fully multi-discrete actions. `acceleration_steps` and `steer_steps` (default
  to 5) control the number of discrete values for acceleration and steering
  respectively.
- `supertuxkart/flattened_discrete-v0` (wrappers: `ConstantSizedObservations`, `PolarObservations`, `DiscreteActionsWrapper`, `FlattenerWrapper`, `FlattenMultiDiscreteActions`) is like the previous one, but with fully
  discretized actions

The reward $r_t$ at time $t$ is given by

$$ r_{t} =  \frac{1}{10}(d_{t} - d_{t-1}) + (1 - \frac{\mathrm{pos}_t}{K})
\times (3 + 7 f_t) - 0.1 + 10 * f_t $$

where $d_t$ is the overall track distance at time $t$, $\mathrm{pos}_t$ the
position among the $K$ karts at time $t$, and $f_t$ is $1$ when the kart
finishes the race.

## Wrappers

Wrappers can be used to modify the environment.

### Constant-size observation

`pystk2_gymnasium.ConstantSizedObservations( env, state_items=5,
  state_karts=5, state_paths=5 )` ensures that the number of observed items,
karts and paths is constant. By default, the number of observations per category
is 5.

### Polar observations

`pystk2_gymnasium.PolarObservations(env)` changes Cartesian
coordinates to polar ones (angle in the horizontal plane, angle in the vertical plan, and distance) of all 3D vectors.

### Discrete actions

`pystk2_gymnasium.DiscreteActionsWrapper(env, acceleration_steps=5, steer_steps=7)` discretizes acceleration and steer actions (5 and 7 values respectively).

### Flattener (actions and observations)

This wrapper groups all continuous and discrete spaces together.

`pystk2_gymnasium.FlattenerWrapper(env)` flattens **actions and
observations**. The base environment should be a dictionary of observation
spaces. The transformed environment is a dictionary made with two entries,
`discrete` and `continuous` (if both continuous and discrete
observations/actions are present in the initial environment, otherwise it is
either the type of `discrete` or `continuous`). `discrete` is `MultiDiscrete`
space that combines all the discrete (and multi-discrete) observations, while
`continuous` is a `Box` space.

### Flatten multi-discrete actions

`pystk2_gymnasium.FlattenMultiDiscreteActions(env)` flattens a multi-discrete
action space into a discrete one, with one action per possible unique choice of
actions. For instance, if the initial space is $\{0, 1\} \times \{0, 1, 2\}$,
the action space becomes $\{0, 1, \ldots, 6\}$.


## Multi-agent environment

`supertuxkart/multi-full-v0` can be used to control multiple karts. It takes an
`agents` parameter that is a list of `AgentSpec`. Observations and actions are a
dictionary of single-kart ones where **string** keys that range from `0` to
`n-1` with `n` the number of karts.

To use different gymnasium wrappers, one can use a `MonoAgentWrapperAdapter`.

Let's look at an example to illustrate this:

```py

from pystk_gymnasium import AgentSpec

agents = [
    AgentSpec(use_ai=True, name="Yin Team", camera_mode=CameraMode.ON),
    AgentSpec(use_ai=True, name="Yang Team", camera_mode=CameraMode.ON),
    AgentSpec(use_ai=True, name="Zen Team", camera_mode=CameraMode.ON)
]

wrappers = [
    partial(MonoAgentWrapperAdapter, wrapper_factories={
        "0": lambda env: ConstantSizedObservations(env),
        "1": lambda env: PolarObservations(ConstantSizedObservations(env)),
        "2": lambda env: PolarObservations(ConstantSizedObservations(env))
    }),
]

make_stkenv = partial(
    make_env,
    "supertuxkart/multi-full-v0",
    render_mode="human",
    num_kart=5,
    agents=agents,
    wrappers=wrappers
)
```

## CLI Commands

The `pystk2` command-line tool provides commands for running races locally or in a distributed client-server setup.

### `pystk2 race` — Run a local race

Runs a race with one or more agents loaded locally.

```bash
pystk2 race agent1.zip agent2.zip --num-karts 5 --track lighthouse --laps 2
```

**Positional arguments:**
- `agents` — One or more agent sources: path to a zip file, a directory containing `pystk_actor.py`, or a Python module name. Append `@:Name` to override the player name (e.g. `agent.zip@:Alice`).

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--num-karts` | 3 | Total number of karts in the race |
| `--track` | random | Track name |
| `--laps` | 1 | Number of laps |
| `--max-paths` | unlimited | Maximum path nodes ahead in observations |
| `--output FILE` | — | Write JSON race results to file |
| `--error-handling` | `raise` | `raise` to propagate agent errors, `catch` to use random actions |
| `--action-timeout` | none | Per-action timeout in seconds (Unix only) |
| `--hide` | off | Run without graphics (headless) |
| `--web` | off | Enable web visualization dashboard (requires `dash`/`plotly`) |
| `--web-port` | 8050 | Port for the web dashboard |
| `--record FILE` | — | Save race video (e.g. `race.mp4`) |
| `--cameras` | auto | Number of cameras (max 8) |
| `--fps` | 20 | Video frame rate |
| `--adapter PATH` | — | Python file providing a custom `create_actor` function |
| `--max-steps` | unlimited | Maximum steps before stopping |

### `pystk2 race-server` — Start an agent server

Starts a persistent server that loads agents and responds to action requests from a race client over ZMQ. The server stays alive across multiple races until interrupted with Ctrl+C.

```bash
# Serve one agent
pystk2 race-server my_agent.zip

# Serve multiple agents on a custom port
pystk2 race-server agent_a.zip agent_b/ --address tcp://*:5556
```

Requires `pyzmq`: `pip install pystk2-gymnasium[remote]`

**Positional arguments:**
- `agents` — One or more agent sources (same format as `pystk2 race`).

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--address` | `tcp://*:5555` | ZMQ bind address |
| `--adapter PATH` | — | Python file providing a custom `create_actor` function |
| `--action-timeout` | none | Per-action timeout in seconds (Unix only) |
| `--threads` | half CPU cores | Number of worker threads for concurrent client sessions |

### `pystk2 race-client` — Run a race with remote servers

Connects to one or more race servers, runs the STK environment locally, sends observations to each server and receives actions.

```bash
# Single server
pystk2 race-client --server tcp://localhost:5555 --num-karts 3 --track lighthouse

# Multiple servers (each student runs their own server)
pystk2 race-client \
  --server tcp://student-a:5555 \
  --server tcp://student-b:5555 \
  --num-karts 5 --track lighthouse --max-steps 500
```

Requires `pyzmq`: `pip install pystk2-gymnasium[remote]`

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--server ADDR` | *(required, repeatable)* | Server address (e.g. `tcp://localhost:5555`) |
| `--num-karts` | 3 | Total number of karts in the race |
| `--track` | random | Track name |
| `--laps` | 1 | Number of laps |
| `--max-paths` | unlimited | Maximum path nodes ahead in observations |
| `--output FILE` | — | Write JSON race results to file |
| `--error-handling` | `raise` | `raise` to propagate agent errors, `catch` to use random actions |
| `--hide` | off | Run without graphics (headless) |
| `--web` | off | Enable web visualization dashboard |
| `--web-port` | 8050 | Port for the web dashboard |
| `--record FILE` | — | Save race video |
| `--cameras` | auto | Number of cameras (max 8) |
| `--fps` | 20 | Video frame rate |
| `--max-steps` | unlimited | Maximum steps before stopping |
| `--timeout` | 60 | ZMQ recv timeout per request in seconds |

### Client-server architecture

The client-server mode is designed for settings where each participant runs their own agent server and a race organizer runs the client:

```
Student A (server)           Student B (server)           Organizer (client)
pystk2 race-server           pystk2 race-server           pystk2 race-client
  agent_a.zip                  agent_b.zip                  --server A:5555
  --address tcp://*:5555       --address tcp://*:5555       --server B:5555
```

Key design points:

- **Wrappers applied server-side**: The client sends raw observations from the base `supertuxkart/multi-full-v0` environment. Each server builds the full wrapper chain (registered wrappers from `env_name` + agent's `get_wrappers()`) and applies them before calling actors, then un-wraps actions before returning them to the client.
- **Persistent servers**: Servers stay alive between races. After a client sends `CLOSE`, the server returns to waiting for the next `INIT`.
- **Concurrent sessions**: The server uses a thread pool (`--threads`) to handle multiple client races simultaneously.
- **One server = one or more agents**: A single server can load multiple agents.
- **Protocol**: ZMQ ROUTER/REQ over TCP with pickle serialization.

## Action and observation space

All the 3D vectors are within the kart referential (`z` front, `x` left, `y`
up):

- `distance_down_track`: The distance from the start
- `energy`: remaining collected energy
- `front`: front of the kart (3D vector)
- `attachment`: the item attached to the kart (bonus box, banana, nitro/big,
  nitro/small, bubble gum, easter egg)
- `attachment_time_left`: how much time the attachment will be kept
- `items_position`: position of the items (3D vectors)
- `items_type`: type of the item
- `jumping`: is the kart jumping
- `karts_position`: position of other karts, beginning with the ones in front
- `max_steer_angle` the max angle of the steering (given the current speed)
- `center_path_distance`: distance to the center of the path
- `center_path`: vector to the center of the path
- `paths_start`, `paths_end`, `paths_width`: 3D vectors to the paths start and
  end, and vector of their widths (scalar). The paths are sorted so that the
  first element of the array is the current one.
- `paths_distance`: the distance of the paths starts and ends (vector of
  dimension 2)
- `powerup`: collected power-up
- `shield_time`
- `skeed_factor`
- `velocity`: velocity vector

## Example

```py3
import gymnasium as gym
from pystk2_gymnasium import AgentSpec


# STK gymnasium uses one process
if __name__ == '__main__':
  # Use a a flattened version of the observation and action spaces
  # In both case, this corresponds to a dictionary with two keys:
  # - `continuous` is a vector corresponding to the continuous observations
  # - `discrete` is a vector (of integers) corresponding to discrete observations
  env = gym.make("supertuxkart/flattened-v0", render_mode="human", agent=AgentSpec(use_ai=False))

  ix = 0
  done = False
  state, *_ = env.reset()

  while not done:
      ix += 1
      action = env.action_space.sample()
      state, reward, terminated, truncated, _ = env.step(action)
      done = truncated or terminated

  # Important to stop the STK process
  env.close()
```
