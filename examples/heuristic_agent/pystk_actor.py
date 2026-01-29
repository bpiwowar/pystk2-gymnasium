"""Heuristic STK agent: follows the track, fires items, always uses nitro."""

import math

import numpy as np

# Environment with fixed-size observations + polar coordinates
env_name = "supertuxkart/simple-v0"
player_name = "Heuristic"


def get_actor(state, observation_space, action_space):
    """Return an actor callable: observation -> action.

    The heuristic:
    - Steers towards the first path endpoint (polar angle)
    - Always accelerates at full throttle
    - Always uses nitro
    - Fires items when carrying a powerup
    - Never brakes, drifts, or calls rescue
    """

    def actor(obs):
        # paths_end is shape (N, 3) with polar coords (angle_zx, angle_zy, dist)
        # after PolarObservations wrapper
        paths_end = obs["paths_end"]
        if len(paths_end) > 0:
            # Steer towards first path endpoint using the ZX-plane angle
            angle_zx = float(paths_end[0][0])
            steer = np.clip(angle_zx / math.pi * 2, -1.0, 1.0)
        else:
            steer = 0.0

        # Fire when we have a powerup (attachment != NOTHING which is last enum)
        attachment = int(obs.get("attachment", 0))
        fire = 1 if attachment != 0 else 0

        return {
            "acceleration": np.array([1.0], dtype=np.float32),
            "steer": np.array([steer], dtype=np.float32),
            "brake": 0,
            "drift": 0,
            "fire": fire,
            "nitro": 1,
            "rescue": 0,
        }

    return actor
