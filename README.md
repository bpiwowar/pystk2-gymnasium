# PySuperTuxKart gymnasium wrapper

*warning*: pystk2-gymnasium is in alpha stage - the environments might change abruptly!

## Environments

After importing `pystk2_gymnasium`, the following environments are available:

- `supertuxkart-v0` is the main environment containing complete observations
- `supertuxkart-simple-v0` is a simplified environment with fixed size observations
- `supertuxkart-flattened-v0` has observation and action spaces simplified at the maximum (only `discrete` and `continuous` keys)

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
