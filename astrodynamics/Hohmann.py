import numpy as np
import math
from . import oe2rv

def twoImpulseHohmann(r1: float, r2: float, i1: float, i2: float, Omega1: float, Omega2: float, mu: float) -> tuple[float, float]:
    """
    Computes the delta-v values for a two-impulse Hohmann-type transfer between two circular orbits.
    
    This function computes the delta v and the orbit transfer associated with
    a two-impulse Hohmann-type transfer with an inclination change. 

    Args
    ----
    `r1` (float): Radius of the initial circular orbit
    
    `r2` (float): Radius of the terminal circular orbit
    
    `i1` (float): Inclination of the initial orbit [rad]
    
    `i2` (float): Inclination of the terminal orbit [rad]
    
    `Omega1` (float): Longitude of the ascending node Ω of the initial orbit [rad]
    
    `Omega2` (float): Longitude of the ascending node Ω of the terminal orbit [rad]
    
    `mu` (float): µ, gravitational parameter of the central body

    Returns
    -------
    tuple: A tuple containing two floats:
        `dv1`: Delta-v magnitude for the first impulse
        
        `dv2`: Delta-v magnitude for the second impulse
    
    Example
    -------
    >>> r1 = 6728.145
    >>> i1 = 0.5
    >>> Omega1 = 0
    >>> r2 = 26558
    >>> i2 = 1
    >>> Omega2 = 1.57
    >>> mu = 398600
    >>> dv1, dv2 = twoImpulseHohmann(r1, r2, i1, i2, Omega1, Omega2, mu)
    >>> print(dv1)
    2.0260453396389324
    >>> print(dv2)
    3.4670421143373877
    """
    
    # Input validation
    if r1 <= 0:
        raise ValueError('The initial radius must be positive')
    if r2 <= 0:
        raise ValueError('The terminal radius must be positive')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    # Normalize angles
    twoPi = 2 * math.pi
    Omega1 = Omega1 % twoPi
    Omega2 = Omega2 % twoPi
    i1 = i1 % twoPi
    if i1 > math.pi:
        i1 -= twoPi
    i2 = i2 % twoPi
    if i2 > math.pi:
        i2 -= twoPi
    
    # Initial orbit
    oe1 = np.array([r1, 0, Omega1, i1, 0, 0])
    r1PCI, v1PCI = oe2rv.oe2rv(oe1, mu)
    h1 = np.cross(r1PCI, v1PCI)
    
    # Terminal orbit
    oe2 = np.array([r2, 0, Omega2, i2, 0, 0])
    r2PCI, v2PCI = oe2rv.oe2rv(oe2, mu)
    h2 = np.cross(r2PCI, v2PCI)
    
    # Impulse 1 (pure energy change at periapsis of transfer orbit)
    dv1 = math.sqrt(mu / r1) * (math.sqrt((2 * r2) / (r1 + r2)) - 1)
    
    # Impulse 2 (energy change and orbit crank at apoapsis of transfer orbit)
    v2minus = math.sqrt(mu) * math.sqrt(2 / r2 - 2 / (r1 + r2))
    v2plus = math.sqrt(mu / r2)
    theta = math.acos(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2)))
    dv2 = math.sqrt(v2minus**2 + v2plus**2 - 2 * v2minus * v2plus * math.cos(theta))
    
    return dv1, dv2


def twoNImpulseOrbitTransfer(r_i: float, r_f: float, i_i: float, i_f: float, Omega_i: float, Omega_f: float, N: int, mu: float) -> tuple[np.ndarray[float], np.ndarray[float], float]:
    """
    Computes the total delta-v and periapsis/apoapsis impulses for a 2N-impulse transfer.
    
    This function computes the total delta-v and the orbit transfer associated with a 2N-impulse
    transfer, where N is the number of periapsis/apoapsis impulse pairs.

    Args
    ----
    `r_i` (float): Radius of the initial circular orbit
    
    `r_f` (float): Radius of the terminal circular orbit
    
    `i_i` (float): Inclination of the initial orbit [rad]
    
    `i_f` (float): Inclination of the terminal orbit [rad]
    
    `Omega_i` (float): Longitude of the ascending node Ω of the initial orbit [rad]
    
    `Omega_f` (float): Longitude of the ascending node Ω of the terminal orbit [rad]
    
    `N` (int): Number of periapsis/apoapsis impulse pairs
    
    `mu` (float): µ, gravitational parameter of the central body
    
    Returns
    -------
    tuple: A tuple containing three objects:
        `dvp` (numpy.ndarray): Array of delta-v magnitudes for `N` periapsis impulses
        
        `dva` (numpy.ndarray): Array of delta-v magnitudes for `N` apoapsis impulses
        
        `dV` (float): Total delta-v for the entire orbit transfer
    
    Example
    -------
    >>> r_i = 6728.145
    >>> i_i = 0.5
    >>> Omega_i = 0
    >>> r_f = 26558
    >>> i_f = 1
    >>> Omega_f = 1.57
    >>> N = 3
    >>> mu = 398600
    >>> dvp, dva, dV = twoNImpulseOrbitTransfer(r_i, r_f, i_i, i_f, Omega_i, Omega_f, N, mu)
    >>> print(dvp)
    [1.17763148 0.51822055 0.3071204 ]
    >>> print(dva)
    [1.89747927 1.70492797 1.66871967]
    >>> print(dV)
    7.2740993341237274
    """
    
    # Input validation
    if r_i <= 0:
        raise ValueError('The initial radius must be positive')
    if r_f <= 0:
        raise ValueError('The terminal radius must be positive')
    if not isinstance(N, int):
        raise TypeError('Number of periapsis/apoapsis impulse pairs must be an integer')
    if N <= 0:
        raise ValueError('Number of periapsis/apoapsis impulse pairs must be positive')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    # Initialize variables
    dV = 0
    r_n = np.zeros(N)
    Omega_n = np.zeros(N)
    i_n = np.zeros(N)
    dvp = np.zeros(N)
    dva = np.zeros(N)
    
    # Calculate total transfer deltaV
    for n in range(N):
        r_n[n] = r_i + ((n + 1) / N) * (r_f - r_i)
        Omega_n[n] = Omega_i + ((n + 1) / N) * (Omega_f - Omega_i)
        i_n[n] = i_i + ((n + 1) / N) * (i_f - i_i)
        
        if n == 0:
            dvp[n], dva[n] = twoImpulseHohmann(r_i, r_n[n], i_i, i_n[n], Omega_i, Omega_n[n], mu)
        elif n == N:
            dvp[n], dva[n] = twoImpulseHohmann(r_n[n-1], r_f, i_n[n-1], i_f, Omega_n[n-1], Omega_f, mu)
        else:
            dvp[n], dva[n] = twoImpulseHohmann(r_n[n-1], r_n[n], i_n[n-1], i_n[n], Omega_n[n-1], Omega_n[n], mu)
        
        # Calculate total delta V
        dV += (dvp[n] + dva[n])
    
    return dvp, dva, dV


def biEllipticWithPlaneChange(oe0: np.ndarray[float], oef: np.ndarray[float], S: float, f: float, mu: float) -> tuple[float, float, float, float]:
    """
    Computes the delta-v values for a three-impulse bi-elliptic-type transfer between two circular orbits.
    
    This function computes the delta v and the orbit transfer associated with a
    three-impulse bi-elliptic-type transfer with an inclination change.

    Args
    ----
    `oe0` (numpy.ndarray): Classical orbital elements of the initial circular orbit
    
    `oef` (numpy.ndarray): Classical orbital elements of the terminal circular orbit
    
    `S` (float): Ratio ri/rf that defines the ratio of the intermediate apoapsis radius to the radius of the terminal orbit
    
    `f` (float): Fraction of the orbit crank performed at the first impulse
    
    `mu` (float): µ, gravitational parameter of the central body
    
    Returns
    -------
    tuple: A tuple containing four objects:
        `dv1` (float): Magnitude of the impulse applied at the periapsis of the first transfer orbit
        
        `dv2` (float): Magnitude of the impulse applied at the apoapsis of the first transfer orbit
        
        `dv3` (float): Magnitude of the impulse applied at the periapsis of the second transfer orbit
        
        `dV` (float): Total delta-v for the entire orbit transfer
    
    Example
    -------
    >>> import numpy as np
    >>> oe0 = np.array([6728.145, 0, np.radians(60), np.radians(28), 0, 0])
    >>> oef = np.array([26578.145, 0, np.radians(120), np.radians(57), 0, 0])
    >>> S = 10
    >>> f = 0.25
    >>> mu = 398600
    >>> dv1, dv2, dv3, dV = biEllipticWithPlaneChange(oe0, oef, S, f, mu)
    >>> print(dv1)
    3.582917451056039
    >>> print(dv2)
    0.3396367687956478
    >>> print(dv3)
    3.502928123295904
    >>> print(dV)
    7.425482343147591
    """
    
    # Input validation
    if not isinstance(oe0, np.ndarray) or not isinstance(oef, np.ndarray):
        raise TypeError('Input orbital elements must be provided as a numpy.ndarray')
    if oe0.shape != (6,) or oef.shape != (6,):
        raise ValueError('Input orbital elements must be a 6-element numpy.ndarray: shape is not (6,)')
    if oe0[0] <= 0 or oef[0] <= 0:
        raise ValueError('Semi-major axis must be a positive value')
    if oe0[1] != 0 or oef[1] != 0:
        raise ValueError('Orbit(s) are not circular')
    if not 0 <= f <= 1:
        raise ValueError('Fraction f must be between 0 and 1, inclusive')
    if S <= 0:
        raise ValueError('Ratio S must be greater than 0')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    # Initial orbit
    r0PCI, v0PCI = oe2rv.oe2rv(oe0, mu)
    h0 = np.cross(r0PCI, v0PCI)

    # Terminal orbit
    rfPCI, vfPCI = oe2rv.oe2rv(oef, mu)
    hf = np.cross(rfPCI,vfPCI)

    # Transfer orbit
    r0 = np.linalg.norm(r0PCI)
    rf = np.linalg.norm(rfPCI)
    ri = S * rf

    theta = math.acos(np.dot(h0, hf) / (np.linalg.norm(h0) * np.linalg.norm(hf)))
    # Impulse 1
    v1minus = math.sqrt(mu / r0)
    v1plus = math.sqrt(mu) * math.sqrt(2 / r0 - 2 / (r0 + ri))
    dv1 = math.sqrt(v1minus**2 + v1plus**2 - 2 * v1minus * v1plus * math.cos(f * theta))

    # Impulse 2
    v2minus = math.sqrt(mu) * math.sqrt(2 / ri - 2 / (r0 + ri))
    v2plus = math.sqrt(mu) * math.sqrt(2 / ri - 2 / (rf + ri))
    dv2 = math.sqrt(v2minus**2 + v2plus**2 - 2 * v2minus * v2plus * math.cos((1 - f) * theta))

    # Impulse 3
    dv3 = math.sqrt(mu / rf) * math.sqrt((2 * ri) / (rf + ri) - 1)
    
    return dv1, dv2, dv3, dv1 + dv2 + dv3


if __name__ == '__main__':
    # Testing
    import time
    #import cProfile
    r1 = 6728.145
    i1 = 0.5
    Omega1 = 0
    r2 = 26558
    i2 = 1
    Omega2 = 1.57
    mu = 398600
    oper = 1000
    t0 = time.perf_counter()
    for _ in range(oper):
        dv1, dv2 = twoImpulseHohmann(r1, r2, i1, i2, Omega1, Omega2, mu)
    t1 = time.perf_counter()
    assert math.isclose(dv1, 2.0260453396389324)
    assert math.isclose(dv2, 3.4670421143373877)
    print(f'Completed twoImpulse in {(t1-t0)*1e3:0.5f} ms')
    print(f'twoImpulse: {oper/(t1-t0):.0f} operations/second\n')
    #cProfile.run('twoImpulseHohmann(r1, r2, i1, i2, Omega1, Omega2, mu)')
    
    r_i = 6728.145
    i_i = 0.5
    Omega_i = 0
    r_f = 26558
    i_f = 1
    Omega_f = 1.57
    N = 3
    mu = 398600
    t0 = time.perf_counter()
    for _ in range(oper):
        dvp, dva, dV = twoNImpulseOrbitTransfer(r_i, r_f, i_i, i_f, Omega_i, Omega_f, N, mu)
    t1 = time.perf_counter()
    assert np.allclose(dvp, np.array([1.17763148, 0.51822055, 0.3071204]))
    assert np.allclose(dva, np.array([1.89747927, 1.70492797, 1.66871967]))
    assert math.isclose(dV, 7.2740993341237274)
    print(f'Completed twoNImpulse in {(t1-t0)*1e3:0.5f} ms')
    print(f'twoNImpulse: {oper/(t1-t0):.0f} operations/second\n')
    #cProfile.run('twoNImpulseOrbitTransfer(r_i, r_f, i_i, i_f, Omega_i, Omega_f, N, mu)')
    
    oe0 = np.array([6728.145, 0, math.radians(60), math.radians(28), 0, 0])
    oef = np.array([26578.145, 0, math.radians(120), math.radians(57), 0, 0])
    S = 10
    f = 0.25
    mu = 398600
    t0 = time.perf_counter()
    for _ in range(oper):
        dv1, dv2, dv3, dV = biEllipticWithPlaneChange(oe0, oef, S, f, mu)
    t1 = time.perf_counter()
    assert math.isclose(dv1, 3.582917451056039)
    assert math.isclose(dv2, 0.3396367687956478)
    assert math.isclose(dv3, 3.502928123295904)
    assert math.isclose(dV, 7.425482343147591)
    print(f'Completed biElliptic in {(t1-t0)*1e3:0.5f} ms')
    print(f'biElliptic: {oper/(t1-t0):.0f} operations/second')
    #cProfile.run('biEllipticWithPlaneChange(oe0, oef, S, f, mu)')