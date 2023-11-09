from typing import Type
import numpy as np
import gymnasium.spaces as spaces

# Tries to use numba if it exists
try:
    from numba import jit
except ImportError:

    class jit:
        def __init__(self, *args, **kwargs):
            # Just ignores
            pass

        def __call__(self, method):
            return method


@jit(nopython=True)
def rotate(v: np.array, q: np.array):
    """Compute the rotation of vector with a quaternion

    Formula from
    http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/transforms/derivations/vectors/index.htm

    :param v: The 3D vector
    :param q: The 4D quaternion
    :return: The vector after rotation
    """
    x, y, z = v
    qw, qx, qy, qz = q

    return np.array(
        [
            x * (qx * qx + qw * qw - qy * qy - qz * qz)
            + y * (2 * qx * qy - 2 * qw * qz)
            + z * (2 * qx * qz + 2 * qw * qy),
            x * (2 * qw * qz + 2 * qx * qy)
            + y * (qw * qw - qx * qx + qy * qy - qz * qz)
            + z * (-2 * qw * qx + 2 * qy * qz),
            x * (-2 * qw * qy + 2 * qx * qz)
            + y * (2 * qw * qx + 2 * qy * qz)
            + z * (qw * qw - qx * qx - qy * qy + qz * qz),
        ],
        dtype=v.dtype,
    )


def max_enum_value(EnumType: Type):
    """Returns the maximum enum value in a given enum type"""
    return max([v.value for v in EnumType.Type.__members__.values()]) + 1


class Discretizer:
    def __init__(self, space: spaces.Box, values: int):
        self.max_value = float(space.high)
        self.min_value = float(space.low)
        self.values = values
        self.space = spaces.Discrete(values)

    def discretize(self, value: float):
        v = int(
            (value - self.min_value)
            * (self.values - 1)
            / (self.max_value - self.min_value)
        )
        assert v >= 0, f"discretized value {v} is below 0"
        if v >= self.values:
            v -= 1
        assert v <= self.values, f"discretized value {v} is above {self.values-1}"
        return v

    def continuous(self, value: int):
        return (self.max_value - self.min_value) * value / (
            self.values - 1
        ) + self.min_value
