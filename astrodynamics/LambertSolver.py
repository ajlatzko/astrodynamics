import numpy as np
import warnings
import math
import cmath

def LambertSolver(pos1: np.ndarray[float], pos2: np.ndarray[float], dt: float, mu: float) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Solves Lambert's problem from two positions.
    
    This function solves Lambert's problem utilizing Lagrange coefficients.

    Args
    ----
    `pos1` (numpy.ndarray): Initial position vector `[x0, y0, z0]` [m]
    
    `pos2` (numpy.ndarray): Final position vector `[x1, y1, z1]`
    
    `dt` (float): Time of flight for the transfer orbit [s]
    
    `mu` (float): Gravitational parameter of the central body

    Returns
    -------
    tuple: A tuple containing the velocity vectors of the transfer orbit at the initial and final positions:
        `v1` (numpy.ndarray): The velocity vector of the transfer orbit at the initial position
        
        `v2` (numpy.ndarray): The velocity vector of the transfer orbit at the final position
    """
    
    # Input validation
    if not isinstance(pos1, np.ndarray) or not isinstance(pos2, np.ndarray):
        raise TypeError('Input positions must be provided as a numpy.ndarray')
    if pos1.shape != (3,) or pos2.shape != (3,):
        raise ValueError('Input positions must be a 3-element numpy.ndarray: shape is not (3,)')
    if dt <= 0:
        raise ValueError('The time of flight must be positive')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    # Compute magnitudes of pos1 and pos2
    r1 = np.linalg.norm(pos1)
    r2 = np.linalg.norm(pos2)

    c12 = np.cross(pos1, pos2)
    theta = math.acos(np.dot(pos1, pos2) / (r1 * r2))

    if c12[2] <= 0:
        theta = 2*np.pi - theta

    A = math.sin(theta) * math.sqrt(r1*r2 / (1 - math.cos(theta)))

    # Determine approximately where F(z,t) changes sign, and
    # use that value of z as the starting value for Equation 5.45:
    z = -100
    while _Fun(z, dt, r1, r2, A, mu).real < 0:
        z += 5
    
    # NEWTONS METHOD TO DETERMINE z
    # Set an error tolerance and a limit on the number of iterations:
    tol = 1.0e-8
    nmax = 5000

    # Iterate on Equation 5.45 until z is determined to within the error tolerance:
    ratio = 1
    n = 0
    while abs(ratio) > tol and n <= nmax:
        n += 1
        ratio = _Fun(z, dt, r1, r2, A, mu) / _dFdz(z, r1, r2, A)
        z -= ratio
    
    # Report if the maximum number of iterations is exceeded:
    if n >= nmax:
        warnings.warn(f'Number of iterations exceeds {nmax} in ast.LambertSolver', RuntimeWarning)

    # DETERMINE LAGRANGE COEFFICIENTS
    # Equation 5.46a:
    f = 1 - _yFun(z, r1, r2, A) / r1
    # Equation 5.46b:
    g = A * cmath.sqrt(_yFun(z, r1, r2, A) / mu)
    # Equation 5.46d:
    gdot = 1 - _yFun(z, r1, r2, A) / r2

    # DETERMINE VELOCITIES
    # Equation 5.28:
    v1 = 1/g * (pos2 - f*pos1)
    # Equation 5.29:
    v2 = 1/g * (gdot*pos2 - pos1)
    
    return v1, v2


# Helper functions
def _yFun(z, r1, r2, A):
    return r1 + r2 + A * (z*_stumpffFuncsS(z) - 1) / (math.sqrt(_stumpffFuncsC(z)))


def _Fun(z, t, r1, r2, A, mu):
    yz = _yFun(z, r1, r2, A)
    if yz < 0:
        return cmath.sqrt((yz / _stumpffFuncsC(z))**3) * _stumpffFuncsS(z) + A*cmath.sqrt(yz) - (math.sqrt(mu))*t
    else:
        return (yz / _stumpffFuncsC(z))**1.5 * _stumpffFuncsS(z) + A*(math.sqrt(yz)) - (math.sqrt(mu))*t


def _dFdz(z, r1, r2, A):
    if z == 0:
        y0 = _yFun(0, r1, r2, A)
        return (math.sqrt(2))/40 * (y0**1.5) + A/8 * ((math.sqrt(y0)) + A*math.sqrt(1 / 2 / y0))
    else:
        yz = _yFun(z, r1, r2, A)
        Cz = _stumpffFuncsC(z)
        Sz = _stumpffFuncsS(z)
        if yz < 0:
            return cmath.sqrt((yz / Cz)**3) * (1 / 2 / z * (Cz - 3*Sz/2/Cz) + 3 * Sz**2 / 4 / Cz) + A/8 * (3 * Sz / Cz * cmath.sqrt(yz) + A*cmath.sqrt(Cz / yz))
        else:
            return (yz / Cz)**1.5 * (1 / 2 / z * (Cz - 3*Sz/2/Cz) + 3 * Sz**2 / 4 / Cz) + A/8 * (3 * Sz / Cz * (math.sqrt(yz)) + A*math.sqrt(Cz / yz))


def _stumpffFuncsC(z):
    if z > 0:
        return (1 - math.cos(math.sqrt(z))) / z
    elif z < 0:
        return (math.cosh(math.sqrt(-z)) - 1) / (-z)
    else:
        return 0.5


def _stumpffFuncsS(z):
    if z > 0:
        return (math.sqrt(z) - math.sin(math.sqrt(z))) / (z**1.5)
    elif z < 0:
        return (math.sinh(math.sqrt(-z)) - math.sqrt(-z)) / ((-z)**1.5)
    else:
        return 1 / 6


if __name__ == '__main__':
    # Testing
    import time
    #import cProfile
    pos1 = np.array([-15000, 10000, -200])
    pos2 = np.array([10000, 15000, 100])
    #r1 = np.linalg.norm(pos1)
    #r2 = np.linalg.norm(pos2)
    #theta = math.acos(np.dot(pos1, pos2) / (r1 * r2))
    #A = math.sin(theta) * math.sqrt(r1*r2 / (1 - math.cos(theta)))
    t = 3600
    mu = 398600
    #y0 = yFun(0, r1, r2, A)
    oper = 10000
    t0 = time.perf_counter()
    for i in range(oper):
        v1, v2 = LambertSolver(pos1, pos2, t, mu)
    t1 = time.perf_counter()
    print(f'Completed in {(t1-t0)*1e3:0.5f} ms')
    print(f'{oper/(t1-t0):.0f} operations/second')
    #cProfile.run('LambertSolver(pos1, pos2, t, mu)')
    