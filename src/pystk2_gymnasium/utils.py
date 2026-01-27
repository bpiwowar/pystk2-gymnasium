from enum import Enum
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


@jit(nopython=True)
def rotate_batch(vectors: np.ndarray, q: np.array) -> np.ndarray:
    """Rotate multiple vectors by a quaternion (vectorized, JIT-compiled).

    :param vectors: Array of shape (N, 3) containing N 3D vectors
    :param q: The 4D quaternion [qw, qx, qy, qz]
    :return: Array of shape (N, 3) with rotated vectors
    """
    qw, qx, qy, qz = q

    # Pre-compute quaternion terms
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    # Rotation matrix elements (from quaternion)
    r00 = qx2 + qw2 - qy2 - qz2
    r01 = 2 * qx * qy - 2 * qw * qz
    r02 = 2 * qx * qz + 2 * qw * qy
    r10 = 2 * qw * qz + 2 * qx * qy
    r11 = qw2 - qx2 + qy2 - qz2
    r12 = -2 * qw * qx + 2 * qy * qz
    r20 = -2 * qw * qy + 2 * qx * qz
    r21 = 2 * qw * qx + 2 * qy * qz
    r22 = qw2 - qx2 - qy2 + qz2

    n = vectors.shape[0]
    result = np.empty((n, 3), dtype=vectors.dtype)

    for i in range(n):
        x = vectors[i, 0]
        y = vectors[i, 1]
        z = vectors[i, 2]
        result[i, 0] = x * r00 + y * r01 + z * r02
        result[i, 1] = x * r10 + y * r11 + z * r12
        result[i, 2] = x * r20 + y * r21 + z * r22

    return result


def max_enum_value(EnumType: Type):
    """Returns the maximum enum value in a given enum type"""
    if not issubclass(EnumType, Enum):
        EnumType = EnumType.Type

    return max([v.value for v in EnumType.__members__.values()]) + 1


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
        assert v <= self.values, f"discretized value {v} is above {self.values - 1}"
        return v

    def continuous(self, value: int):
        return (self.max_value - self.min_value) * value / (
            self.values - 1
        ) + self.min_value
