# Version 0.4.5

- `center_path_distance` is now relative (to indicate left/right of the path)
- `center_path` gives the vector to the center of the track
- Changed the reward to gives some importance to the final ranking
- Better regression tests

# Version 0.4.2

- Fix in formula for `center_path_distance`

# Version 0.4.0

- Multi-agent environment
- Use polar representation instead of coordinates (except for the "full" environment)
- Only two base environments (multi/mono-agent) and wrappers for the rest: this allows races to be organized with different set of wrappers (depending on the agent)
- Added `center_path_distance`
- Allow to change player name and camera mode
- breaking: Agent spec is used for mono-kart environments
