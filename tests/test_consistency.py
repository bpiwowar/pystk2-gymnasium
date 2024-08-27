import gymnasium
import numpy as np
import numpy.testing
import pystk2
from pystk2_gymnasium.utils import Discretizer, rotate
from pystk2_gymnasium.envs import STKRaceEnv


def test_rotation():
    race = None
    try:
        STKRaceEnv.initialize(False)

        config = pystk2.RaceConfig(num_kart=1, track="lighthouse")

        race = pystk2.Race(config)
        world = pystk2.WorldState()
        race.start()
        world.update()

        kart = world.karts[0]
        np.allclose(kart.velocity_lc, rotate(kart.velocity, kart.rotation))
    finally:
        if race is not None:
            race.stop()
            del race


def test_discretizer():
    k = 5

    discretizer = Discretizer(gymnasium.spaces.Box(-1, 1, shape=(1,)), k)
    step = 2.0 / (k - 1)

    for j in range(k):
        assert discretizer.discretize(discretizer.continuous(j)) == j, f"For index {j}"

    for x in np.arange(-1, 1, step):
        xhat = discretizer.continuous(discretizer.discretize(x))
        assert np.abs(xhat - x) < step, f"For value {x} vs {xhat}"
