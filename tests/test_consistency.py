import gymnasium
import numpy as np
import numpy.testing
import pystk2
from pystk2_gymnasium.utils import Discretizer, rotate
from pystk2_gymnasium.envs import STKRaceEnv


def test_rotation():
    env = None
    try:
        env = STKRaceEnv()
        env.initialize(False)
        env.config = pystk2.RaceConfig(num_kart=1, track="lighthouse")
        env.warmup_race()
        world = env.world_update(False)

        kart = world.karts[0]
        np.allclose(kart.velocity_lc, rotate(kart.velocity, kart.rotation))
    finally:
        if env is not None:
            env.close()


def test_discretizer():
    k = 5

    discretizer = Discretizer(gymnasium.spaces.Box(-1, 1, shape=(1,)), k)
    step = 2.0 / (k - 1)

    for j in range(k):
        assert discretizer.discretize(discretizer.continuous(j)) == j, f"For index {j}"

    for x in np.arange(-1, 1, step):
        xhat = discretizer.continuous(discretizer.discretize(x))
        assert np.abs(xhat - x) < step, f"For value {x} vs {xhat}"
