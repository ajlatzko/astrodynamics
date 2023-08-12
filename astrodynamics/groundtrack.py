import numpy as np
import math
from . import plotting
from . import Kepler

def dcmI2E(t: float, OmegaE: float) -> np.ndarray[float]:
    """
    Computes the 3 x 3 transformation matrix from ECI to ECEF coordinates at time t.
    
    This function computes the transformation matrix from Earth-centered inertial Cartesian
    coordinates to Earth-centered Earth-fixed Cartesian coordinates. The transformation matrix
    is of size 3 by 3 and represents the transformation matrix at the time t.

    Args
    ----
    `t` (float): The time at which the transformation is desired [s]
    
    `OmegaE` (float): The rotation rate of the Earth [rad/s]

    Returns
    -------
    `TI2E` (numpy.ndarray): 3 x 3 matrix that transforms a vector expressed in ECI coordinates to ECEF coordinates
    
    Example
    -------
    >>> import numpy as np
    >>> t = 2500
    >>> OmegaE = (2 * np.pi) / 86400
    >>> TI2E = dcmI2E(t, OmegaE)
    >>> print(TI2E)
    [[ 0.98351892  0.18080525  0.        ]
     [-0.18080525  0.98351892  0.        ]
     [ 0.          0.          1.        ]]
    """
    
    return np.array([[ math.cos(OmegaE * t), math.sin(OmegaE * t), 0],
                     [-math.sin(OmegaE * t), math.cos(OmegaE * t), 0],
                     [                    0,                    0, 1]])


def eci2ecef(tValues: np.ndarray[float], rValuesECI: np.ndarray[float], OmegaE: float) -> np.ndarray[float]:
    """
    Transforms spacecraft orbit position from ECI to ECEF coordinates.
    
    This function transforms the values of the position along a spacecraft orbit from Earth-centered
    inertial Cartesian coordinates to Earth-centered Earth-fixed Cartesian coordinates.

    Args
    ----
    `tValues` (numpy.ndarray): Vector of values of time at which the transformation is to be performed [s]
    
    `rValuesECI` (numpy.ndarray): Matrix of position vectors expressed in ECI coordinates and stored ROW-WISE
    
    `OmegaE` (float): The rotation rate of the Earth [rad/s]

    Returns
    -------
    `rValuesECEF` (numpy.ndarray): Matrix of position vectors expressed in ECEF coordinates and stored ROW-WISE
    
    Example
    -------
    >>> import numpy as np
    >>> tValues = np.array([0, 600, 1200])
    >>> rValuesECI = np.array([[-15000, 10000, -200],[10000, 15000, 100],[-10000, 10000, 0]])
    >>> OmegaE = (2 * np.pi) / 86400
    >>> rValuesECEF = eci2ecef(tValues, rValuesECI, OmegaE)
    >>> print(rValuesECEF)
    [[-15000.          10000.           -200.        ]
     [ 10644.7730263   14549.52945007    100.        ]
     [ -9090.38955344  10833.50440839      0.        ]]
    """
    
    # Input validation
    if not isinstance(tValues, np.ndarray) or not isinstance(rValuesECI, np.ndarray):
        raise TypeError('Input vectors must be provided as a numpy.ndarray')
    if tValues.ndim != 1:
        raise ValueError('Input time vector must be one-dimensional')
    if rValuesECI.shape[1] != 3:
        raise ValueError('Input positions matrix must have 3 columns')
    if rValuesECI.shape[0] != tValues.shape[0]:
        raise ValueError('Input time vector and positions matrix must have the same length')
    
    # Initialize output matrix
    rValuesECEF = np.zeros((tValues.size,3))
    
    # Multiply position vector by its respective transformation matrix
    for i in range(tValues.size):
        rValuesECEF[i,:] = dcmI2E(tValues[i], OmegaE) @ rValuesECI[i,:]
    
    return rValuesECEF


def ecef2LonLat(rValuesECEF: np.ndarray[float]) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Computes Earth-relative longitude and geocentric latitude from ECEF position.
    
    This function computes the Earth-relative longitude and geocentric latitude from the Earth-centered
    Earth-fixed position.

    Args
    ----
    
    `rValuesECEF` (numpy.ndarray): Matrix of position vectors expressed in ECEF coordinates and stored ROW-WISE
    
    Returns
    -------
    tuple: A tuple containing the Earth-relative longitude and geocentric latitude:
        `lonE` (numpy.ndarray): Vector containing the values of the Earth-relative longitude [rad]
        
        `lat` (numpy.ndarray): Vector containing the values of the geocentric latitude [rad]
    
    Example
    -------
    >>> import numpy as np
    >>> rValuesECEF = np.array([[-15000, 10000, -200],[10000, 15000, 100],[-10000, 10000, 0]])
    >>> lonE, lat = ecef2LonLat(rValuesECEF)
    >>> print(lonE)
    [2.55359005 0.98279372 2.35619449]
    >>> print(lat)
    [-0.01109355  0.00554695  0.        ]
    """
    
    # Input validation
    if not isinstance(rValuesECEF, np.ndarray):
        raise TypeError('Input must be provided as a numpy.ndarray')
    if rValuesECEF.shape[1] != 3:
        raise ValueError('Input must have 3 columns')
    
    # Compute Earth-relative longitude and geocentric latitude
    lonE = np.arctan2(rValuesECEF[:,1], rValuesECEF[:,0])
    lat = np.arctan2(rValuesECEF[:,2], np.linalg.norm(rValuesECEF[:,0:2], axis=1))
    
    return lonE, lat


def groundtrack(r0: np.ndarray[float], v0: np.ndarray[float], t0: float, mu: float, OmegaE: float, tf: float, tStep: float = 300):
    """
    Produces a groundtrack for the input orbit and time span.
    
    This function generates a groundtrack by mapping the trajectory of an orbit onto an image of the Earth's
    surface for the specified time span and time step.

    Args
    ----
    
    `r0` (numpy.ndarray): Initial position vector in the Earth-centered inertial frame
    
    `v0` (numpy.ndarray): Initial velocity vector in the Earth-centered inertial frame
    
    `t0` (float): Initial time [s]
    
    `mu` (float): µ, gravitational parameter of the Earth
    
    `OmegaE` (float): The rotation rate of the Earth [rad/s]
    
    `tf` (float): Final time [s]
    
    `tStep` (float, optional): Time step [s]. Default is 300 seconds
    
    Returns
    -------
    An image depicting the generated groundtrack of the orbit over the specified time span
    
    Example
    -------
    >>> import numpy as np
    >>> r0 = np.array([-1217.39430415697, -3091.41210822807, -6173.40732877317])
    >>> v0 = np.array([9.88635815507896, -0.446121737099303, -0.890884522967222])
    >>> t0 = 21600
    >>> tf = 86400
    >>> mu = 398600
    >>> OmegaE = (2 * np.pi) / 86400
    >>> groundtrack(r0, v0, t0, mu, OmegaE, tf)
    """
    
    # Input validation
    if not isinstance(r0, np.ndarray) or not isinstance(v0, np.ndarray):
        raise TypeError('Input vectors must be provided as a numpy.ndarray')
    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError('Input vectors must be a 3-element numpy.ndarray: shape is not (3,)')
    if tf <= t0:
        raise ValueError('Final time must be greater than initial time')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    if tStep <= 0:
        raise ValueError('The time step must be positive')
    
    # Create time span from inputs
    tSpan = np.arange(t0, tf+1, tStep)
    
    # Compute position vector for each time
    rValuesECI = np.zeros((tSpan.size, 3))
    rValuesECI[0,:] = r0
    for i, t in enumerate(tSpan[1:]):
        rValuesECI[i+1,:], _ = Kepler.propagateKepler(r0, v0, t0, t, mu)
    
    # Convert ECI coordinates to ECEF coordinates
    rValuesECEF = eci2ecef(tSpan, rValuesECI, OmegaE)
    
    # Convert ECEF coordinates to lat, lon
    lonE, lat = ecef2LonLat(rValuesECEF)
    
    # Plot groundtrack
    plotting.mercatorDisplay(lonE, lat)


if __name__ == '__main__':
    # Testing
    import time
    #import cProfile
    
    r0 = np.array([-1217.39430415697, -3091.41210822807, -6173.40732877317])
    v0 = np.array([9.88635815507896, -0.446121737099303, -0.890884522967222])
    t0 = 21477.579183269
    tau = 43200
    mu = 398600
    OmegaE = (2 * np.pi) / 86400
    
    t0 = time.perf_counter()
    groundtrack(r0, v0, t0, mu, OmegaE, 2 * tau + t0)
    t1 = time.perf_counter()
    print(f'Completed in {(t1-t0)*1e3:0.5f} ms')
    #cProfile.run('groundtrack(r0, v0, t0, mu, OmegaE, 2 * tau + t0)')