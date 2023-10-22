import numpy as np

# Tries to use numba if it exists
try:
    from numba import jit
except ImportError:

    class jit:
        def __init__(self, *args, **kwargs):
            # Just ignores
            pass


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
