"""Regression tests for the progress used in the reward.

STK's ``overall_distance`` (``finished_laps * L + distance_down_track``) can
jump by a full lap, e.g. when the kart crosses the line off-road (the lap is
only counted once the kart is back on road) or reverses behind the line. The
progress must be computed from the wrapped variations of
``distance_down_track`` instead.
"""

from types import SimpleNamespace

import pytest

from pystk2_gymnasium.envs import BaseSTKRaceEnv

L = 880.0


class FakeSTK:
    def __init__(self, distances):
        self.distances = distances

    def get_world(self):
        return SimpleNamespace(
            karts=[
                SimpleNamespace(distance_down_track=d) for d in self.distances.pop(0)
            ]
        )


def progress(distances):
    """Returns the per-step progress deltas of a single kart"""
    env = object.__new__(BaseSTKRaceEnv)
    env.track = SimpleNamespace(length=L)
    env._stk = FakeSTK([[d] for d in distances])
    env.world_update(False)
    deltas = []
    for _ in range(len(distances) - 1):
        env.world_update()
        deltas.append(env.progress[0] - env.last_progress[0])
    return deltas


def test_forward():
    assert progress([10.0, 20.0, 35.0]) == pytest.approx([10.0, 15.0])


def test_cross_line_forward():
    # Crossing the line: distance_down_track wraps from ~L to ~0
    assert progress([870.0, 878.0, 6.0, 20.0]) == pytest.approx([8.0, 8.0, 14.0])


def test_cross_line_backward():
    # Reversing behind the line must not give a lap worth of progress
    assert progress([5.0, 1.0, 876.0, 870.0]) == pytest.approx([-4.0, -5.0, -6.0])
