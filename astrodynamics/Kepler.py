import numpy as np
import math
import warnings
from . import rv2oe
from . import oe2rv

def KeplerSolver(e: float, M: float) -> float:
    """
    Solves Kepler's equation using a third order method.
    
    This function solves Kepler's equation:
                       E - e*sin(E) = M
    using optimized third-order iteration and starting value methods [1].

    Args
    ----
    `e` (float): Eccentricity of the orbit
    
    `M` (float): The mean anomaly at which to find the eccentric anomaly [rad]
        M = sqrt(mu/a^3)*(t - t0) - 2*pi*k + E0 - e*sin(E0)

    Returns
    -------
    `E` (float): The eccentric anomaly at time t [rad]
    
    Example
    -------
    >>> e = 0.25
    >>> M = 1.57
    >>> E = KeplerSolver(e, M)
    >>> print(E)
    1.8127197442510425
    """
    
    # Input validation
    if not 0 < e < 1:
        if e == 0:
            return M
        else:
            raise ValueError('Eccentricity must be between 0 (inclusive) and 1 (exclusive)')
    
    
    # Define helper functions
    def eps3(e, M, x):
        """
        Returns the third order error in the approximation
        """
        t1 = math.cos(x)
        t2 = -1 + e * t1
        t3 = math.sin(x)
        t4 = e * t3
        t5 = -x + t4 + M
        t6 = t5 / (0.5 * t5 * t4 / t2 + t2)
        return t5 / ((0.5 * t3 - (1/6) * t1 * t6) * e * t6 + t2)
    
    
    def E0_3(e, M):
        """
        Computes the starting approximation for Kepler's
        equation using a third order scheme
        """
        t34 = e ** 2
        t35 = e * t34
        t33 = math.cos(M)
        return M + (-0.5 * t35 + e + (t34 + 1.5 * t33 * t35) * t33) * math.sin(M)
    
    
    M = M % (2 * math.pi)     # Normalize the mean anomaly
    E0 = E0_3(e, M)           # Get initial guess
    tol = np.finfo(float).eps # Set tolerance to machine epsilon
    
    # Employ third order iteration scheme
    dE = tol + 1
    count = 0
    while dE > tol:
        E = E0 - eps3(e, M, E0)
        dE = abs(E - E0)
        E0 = E
        count += 1
        if count == 100:
            warnings.warn(f'Number of iterations exceeds {count} in ast.KeplerSolver', RuntimeWarning)
            break
    
    return E


def propagateKepler(rt0: np.ndarray[float], vt0: np.ndarray[float], t0: float, t: float, mu: float) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Propagates an orbit around a celestial body using Kepler's Equation.
    
    This function computes the PCI position and PCI inertial velocity at t given the
    PCI position and PCI inertial velocity at t0 using Kepler's equation.

    Args
    ----
    `rt0` (numpy.ndarray): Initial position vector in the planet-centered inertial frame
    
    `vt0` (numpy.ndarray): Initial velocity vector in the planet-centered inertial frame
    
    `t0` (float): Initial time
    
    `t` (float): Final time
    
    `mu` (float): µ, gravitational parameter of the central body
    
    Returns
    -------
    tuple: A tuple containing the final position and velocity vectors in the PCI coordinate system:
        `rt` (numpy.ndarray): The position vector in the planet-centered inertial frame at time t
        
        `vt` (numpy.ndarray): The velocity vector in the planet-centered inertial frame at time t
    
    Example
    -------
    >>> import numpy as np
    >>> rt0 = np.array([-5613.97603835865, -2446.44383433555, 2600.48533877841])
    >>> vt0 = np.array([2.12764777374332, -7.13421216656605, -2.1184067703542])
    >>> t0 = 0
    >>> tf = 600
    >>> mu = 398600
    >>> rt, vt = propagateKepler(rt0, vt0, t0, tf, mu)
    >>> print(rt)
    [-3125.47724472 -5815.50982679   821.93420554]
    >>> print(vt)
    [ 5.82776399 -3.63626009 -3.5673933 ]
    """
    
    # Input validation
    if not isinstance(rt0, np.ndarray) or not isinstance(vt0, np.ndarray):
        raise TypeError('Input vectors must be provided as a numpy.ndarray')
    if rt0.shape != (3,) or vt0.shape != (3,):
        raise ValueError('Input vectors must be a 3-element numpy.ndarray: shape is not (3,)')
    if t <= t0:
        raise ValueError('Final time must be greater than initial time')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    twoPi = 2 * math.pi
    
    # Get initial orbital elements
    oe0 = rv2oe.rv2oe(rt0, vt0, mu)
    a = oe0[0]
    e = oe0[1]
    nu0 = oe0[5]
    tau = twoPi * math.sqrt(a**3 / mu)
    
    oneMinusE = math.sqrt(1 - e)
    onePlusE = math.sqrt(1 + e)
    nu0Div2 = nu0 / 2
    
    # Get initial eccentric anomaly
    E0 = 2 * math.atan2(oneMinusE * math.sin(nu0Div2), onePlusE * math.cos(nu0Div2))
    if E0 < 0:
        E0 += twoPi
        
    # Calculate time of periapsis crossing
    tp = t0 - math.sqrt(a**3 / mu) * (E0 - e * math.sin(E0))
    
    # Calculate number of periapsis crossings from tp to t
    k = math.floor((t - tp) / tau)
    
    # Determine mean anomaly at t
    M = math.sqrt(mu / a**3) * (t - t0) - twoPi * k + E0 - e * math.sin(E0)
    
    # Solve Kepler's equation to get eccentric anomaly at t
    E = KeplerSolver(e, M)
    if E < 0:
        E += twoPi
    
    # Get true anomaly at t from E
    EDiv2 = E / 2
    nu = 2 * math.atan2(onePlusE * math.sin(EDiv2), oneMinusE * math.cos(EDiv2))
    if nu < 0:
        nu += twoPi
    
    # Get orbital elements at t and convert to position and inertial velocity
    oe = np.array([a, e, oe0[2], oe0[3], oe0[4], nu])
    rt, vt = oe2rv.oe2rv(oe, mu)
    
    return rt, vt


if __name__ == '__main__':
    # Testing
    import time
    #import cProfile
    #e = 0.25
    M = 1.57
    t0 = time.perf_counter()
    for e in np.arange(0, 1, 0.001):
        E = KeplerSolver(e, M)
    t1 = time.perf_counter()
    print(f'Completed Solver in {(t1-t0)*1e3:0.5f} ms')
    #cProfile.run('KeplerSolver(e, M)')
    
    rt0 = np.array([-5613.97603835865, -2446.44383433555, 2600.48533877841])
    vt0 = np.array([2.12764777374332, -7.13421216656605, -2.1184067703542])
    t0 = 0
    tf = 600
    mu = 398600
    oper = 1000
    to = time.perf_counter()
    for _ in range(oper):
        rt, vt = propagateKepler(rt0, vt0, t0, tf, mu)
    t1 = time.perf_counter()
    print(f'Completed propagation in {(t1-to)*1e3:0.5f} ms')
    print(f'Propagation: {oper/(t1-to):.0f} operations/second')
    #cProfile.run('propagateKepler(rt0, vt0, t0, tf, mu)')