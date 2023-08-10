import numpy as np
import math

def qConcat(q0: np.ndarray[float], q1: np.ndarray[float]) -> np.ndarray[float]:
    """
    Concatenates two quaternions.
    
    This function performs quaternion concatenation by multiplying two quaternions q0 and q1.
    The resulting quaternion represents the composition of rotations encoded by q0 and q1.

    Args
    ----
    `q0` (numpy.ndarray): First quaternion in the form `[w0, x0, y0, z0]`
    
    `q1` (numpy.ndarray): Second quaternion in the form `[w1, x1, y1, z1]`

    Returns
    -------
    `r` (numpy.ndarray): Concatenated quaternion in the form `[w, x, y, z]`
    
    Example
    -------
    >>> import numpy as np
    >>> q0 = np.array([1, 2, 3, 4])
    >>> q1 = np.array([5, 4, 3, 2])
    >>> r = qConcat(q0, q1)
    >>> print(r)
    [-20.   8.  30.  16.]
    """
    
    # Input validation
    if q0.shape != (4,) or q1.shape != (4,):
        raise Exception('Input quaterinon(s) do not have correct shape: (4,)')
    
    # Extract components of the first quaternion
    w0 = q0[0]
    x0 = q0[1]
    y0 = q0[2]
    z0 = q0[3]

    # Extract components of the second quaternion
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]
    
    # Perform quaternion concatenation
    r = np.zeros(4)
    r[0] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    r[1] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    r[2] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    r[3] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return r


def qConj(q: np.ndarray[float]) -> np.ndarray[float]:
    """
    Calculates the conjugate of a quaternion.

    This function calculates the conjugate of the given quaternion by negating the imaginary components.

    Args
    ----
    `q` (numpy.ndarray): Quaternion in the form `[w, x, y, z]`

    Returns
    -------
    numpy.ndarray: Conjugate quaternion in the form `[w, -x, -y, -z]`

    Example
    -------
    >>> import numpy as np
    >>> q = np.array([1, 2, 3, 4])
    >>> r = qConj(q)
    >>> print(r)
    [ 1 -2 -3 -4]
    """
    
    # Input validation
    if q.shape != (4,):
        raise Exception('Input quaterinon does not have correct shape: (4,)')
    
    # Extract quaternion components
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    
    # Calculate conjugate quaternion
    return np.array([w, -x, -y, -z])


def qNorm(q: np.ndarray[float]) -> np.ndarray[float]:
    """
    Normalizes a quaternion.

    This function normalizes the given quaternion by dividing each component by its magnitude.

    Args
    ----
    `q` (numpy.ndarray): Quaternion in the form `[w, x, y, z]`

    Returns
    -------
    numpy.ndarray: Normalized quaternion in the form `[w', x', y', z']`

    Example
    -------
    >>> import numpy as np
    >>> q = np.array([1, 2, 3, 4])
    >>> r = qNorm(q)
    >>> print(r)
    [0.18257419 0.36514837 0.54772256 0.73029674]
    """
    
    # Input validation
    if q.shape != (4,):
        raise Exception('Input quaterinon does not have correct shape: (4,)')
    
    # Calculate the magnitude of the quaternion
    mag = np.linalg.norm(q)

    # Normalize the quaternion by dividing each component by its magnitude
    return np.array([q[0]/mag, q[1]/mag, q[2]/mag, q[3]/mag])


def qFromAngleAxis(angle: float, axis: np.ndarray[float]) -> np.ndarray[float]:
    """
    Constructs a quaternion from an angle-axis representation.

    This function constructs a quaternion from the given angle-axis representation.

    Args
    ----
    `angle` (float): Rotation angle [rad]
    
    `axis` (numpy.ndarray): Axis of rotation in the form `[x, y, z]`

    Returns
    -------
    numpy.ndarray: Quaternion representation of the rotation in the form `[w, x, y, z]`

    Example
    -------
    >>> import numpy as np
    >>> angle = np.pi / 2
    >>> axis = np.array([1, 0, 0])
    >>> q = qFromAngleAxis(angle, axis)
    >>> print(q)
    [0.70710678 0.70710678 0.         0.        ]
    """
    
    # Input validation
    if axis.shape != (3,):
        raise Exception('Axis does not have correct shape: (3,)')
    
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)

    # Calculate half of the angle
    halfAngle = 0.5 * angle

    # Calculate sine and cosine of the half angle
    s = math.sin(halfAngle)
    c = math.cos(halfAngle)

    # Construct the quaternion representation of the rotation
    return qNorm(np.array([c, s*axis[0], s*axis[1], s * axis[2]]))


def qFromToRotation(fr: np.ndarray[float], to: np.ndarray[float]) -> np.ndarray[float]:
    """
    Constructs a quaternion representing a rotation from one vector to another.

    This function constructs a quaternion representing the rotation from the `fr` vector to the `to` vector.

    Args
    ----
    `fr` (numpy.ndarray): Source vector in the form `[x, y, z]`
    
    `to` (numpy.ndarray): Target vector in the form `[x, y, z]`
    
    Returns
    -------
    numpy.ndarray: Quaternion representation of the rotation in the form `[w, x, y, z]`

    Example
    -------
    >>> import numpy as np
    >>> fr = np.array([1, 0, 0])
    >>> to = np.array([0, 1, 0])
    >>> q = qFromToRotation(fr, to)
    >>> print(q)
    [0.70710678 0.         0.         0.70710678]
    """
    
    # Input validation
    if fr.shape != (3,) or to.shape != (3,):
        raise Exception('Input vector(s) do not have correct shape: (3,)')
    
    # Normalize the fr and to vectors
    fr = fr / np.linalg.norm(fr)
    to = to / np.linalg.norm(to)

    # Calculate the dot product between fr and to vectors
    d = np.dot(fr, to)

    if d > (1 - 1e-6):
        # If the vectors are nearly parallel, return identity quaternion
        return np.array([1, 0, 0, 0])
    elif d < -(1 - 1e-6):
        # If the vectors are nearly opposite, rotate 180 degrees around any axis
        return qFromAngleAxis(np.pi, [0, 0, 0, 1])
    else:
        # Calculate the quaternion representing the rotation
        r = np.zeros(4)
        s = math.sqrt(2 * (1 + d))
        invs = 1 / s

        r[0] = 0.5 * s
        r[1] = invs * (fr[1] * to[2] - fr[2] * to[1])
        r[2] = invs * (fr[2] * to[0] - fr[0] * to[2])
        r[3] = invs * (fr[0] * to[1] - fr[1] * to[2])
        return r


def qFromVec(vec: np.ndarray[float]) -> np.ndarray[float]:
    """
    Constructs a quaternion from a vector.

    This function constructs a quaternion from the given vector by setting its scalar component to 0.

    Args
    ----
    `vec` (numpy.ndarray): Input vector in the form `[x, y, z]`
    
    Returns
    -------
    numpy.ndarray: Quaternion representation in the form `[0, x, y, z]`

    Example
    -------
    >>> import numpy as np
    >>> vec = np.array([1, 2, 3])
    >>> q = qFromVec(vec)
    >>> print(q)
    [0 1 2 3]
    """
    
    # Input validation
    if vec.shape != (3,):
        raise Exception('Input vector does not have correct shape: (3,)')
    
    return np.array([0, vec[0], vec[1], vec[2]])


def qToVec(q: np.ndarray[float]) -> np.ndarray[float]:
    """
    Extracts the vector part from a quaternion.

    This function converts the given quaternion to a vector by discarding its real component.

    Args
    ----
    `q` (numpy.ndarray): Input quaternion in the form `[w, x, y, z]`
    
    Returns
    -------
    numpy.ndarray: Extracted vector in the form `[x, y, z]`

    Example
    -------
    >>> import numpy as np
    >>> q = np.array([1, 2, 3, 4])
    >>> vec = qToVec(q)
    >>> print(vec)
    [2 3 4]
    """
    
    # Input validation
    if q.shape != (4,):
        raise Exception('Input quaternion does not have correct shape: (4,)')
    
    return np.array([q[1], q[2], q[3]])


def qRotate(q: np.ndarray[float], vec: np.ndarray[float]) -> np.ndarray[float]:
    """
    Rotates a vector using a quaternion.

    This function rotates the given vector using the specified quaternion.

    Args
    ----
    `q` (numpy.ndarray): Rotation quaternion in the form `[w, x, y, z]`
    
    `vec` (numpy.ndarray): Input vector in the form `[x, y, z]`
    
    Returns
    -------
    numpy.ndarray: Rotated vector in the form `[x', y', z']`

    Example
    -------
    >>> import numpy as np
    >>> vec = np.array([1, 0, 0])
    >>> q = qNorm(np.array([1, 2, 3, 4]))
    >>> rot = qRotate(q, vec)
    >>> print(rot)
    [-0.66666667  0.66666667  0.33333333]
    """
    
    # Input validation
    if q.shape != (4,):
        raise Exception('Input quaternion does not have correct shape: (4,)')
    if vec.shape != (3,):
        raise Exception('Input vector does not have correct shape: (3,)')
    
    # Convert the input vector to a quaternion
    p = qFromVec(vec)

    # Rotate the vector using quaternion multiplication
    return qToVec(qConcat(qConcat(q, p), qConj(q)))