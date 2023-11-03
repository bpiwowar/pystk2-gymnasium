import numpy as np
import pystk2
from pystk2_gymnasium.utils import rotate
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
