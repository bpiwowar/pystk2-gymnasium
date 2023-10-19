import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Tuple, Box, Discrete

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, num_karts=3, num_controlled=1):
        assert num_controlled > 0 and num_controlled <= num_karts

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = Tuple((
            # Acceleration
            Box(0, 1, shape=(1,)),
            # Steering angle
            Box(-1, 1, shape=(1,)),
            # Brake
            Discrete(2),
            # Drift
            Discrete(2),
            # Fire
            Discrete(2),
            # Nitro
            Discrete(2),
            # Call the rescue bird
            Discrete(2),
        ))

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

