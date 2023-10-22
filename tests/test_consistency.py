import numpy as np
import pystk2
from pystk2_gymnasium.utils import rotate


def test_rotation():
    pystk2.init(pystk2.GraphicsConfig.none())

    config = pystk2.RaceConfig(num_kart=1, track="lighthouse")

    race = pystk2.Race(config)
    world = pystk2.WorldState()
    race.start()
    world.update()

    kart = world.karts[0]
    np.allclose(kart.velocity_lc, rotate(kart.velocity, kart.rotation))
