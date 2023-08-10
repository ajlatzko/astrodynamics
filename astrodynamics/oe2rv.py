import numpy as np
import math

def oe2rv(oe: np.ndarray[float], mu: float) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Convert orbital elements to position and velocity vectors in the planet-centered inertial coordinate system.

    Given the six classical orbital elements and the gravitational parameter of the central body,
    this function computes the Cartesian Planet-Centered Inertial (PCI) position and Cartesian
    Planet-Centered Inertial (PCI) velocity vectors for circular and elliptical orbits.

    Args
    ----
    `oe` (numpy.ndarray): Six classical orbital elements in the following order:
        `oe[0]`: Semi-major axis (a)
        
        `oe[1]`: Eccentricity (e)
        
        `oe[2]`: Longitude of the ascending node (Ω, Omega) [rad]
        
        `oe[3]`: Orbital inclination (i) [rad]
        
        `oe[4]`: Argument of periapsis (ω, omega) [rad]
        
        `oe[5]`: True anomaly (ν, nu) [rad]
  
    `mu` (float): µ, gravitational parameter of the central body

    Returns
    -------
    tuple: A tuple containing the position and velocity vectors in the PCI coordinate system:
        `rPCI` (numpy.ndarray): The position vector in the planet-centered inertial frame
        
        `vPCI` (numpy.ndarray): The velocity vector in the planet-centered inertial frame
        
    Example
    -------
    >>> import numpy as np
    >>> oe = np.array([7000, 0.2, 0, 0.8, 0, 1.57])
    >>> mu = 398600
    >>> rPCI, vPCI = oe2rv(oe, mu)
    >>> print(rPCI)
    [   5.35046335 4681.12206183 4819.86376512]
    >>> print(vPCI)
    [-7.70165168  1.07743174  1.10936527]
    """
    
    # Input validation
    if not isinstance(oe, np.ndarray):
        raise TypeError('Input orbital elements must be provided as a numpy.ndarray')
    if oe.shape != (6,):
        raise ValueError('Input orbital elements must be a 6-element numpy.ndarray: shape is not (6,)')
    if oe[0] <= 0:
        raise ValueError('Semi-major axis must be a positive value')
    if not 0 <= oe[1] < 1:
        raise ValueError('Eccentricity must be between 0 (inclusive) and 1 (exclusive)')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    # Get orbital elements from oe vector
    a = float(oe[0])     # Semi-major axis
    e = float(oe[1])     # Eccentricity
    Omega = float(oe[2]) # Longitude of the ascending node
    i = float(oe[3])     # Orbital inclination
    omega = float(oe[4]) # Argument of periapsis
    nu = float(oe[5])    # True anomaly
    
    # Normalize angles
    twoPi = 2 * math.pi
    Omega = Omega % twoPi
    omega = omega % twoPi
    nu = nu % twoPi
    i = i % twoPi
    if i > math.pi:
        i -= twoPi
    
    # Calculate intermediate values
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)
    
    cos_Omega = math.cos(Omega)
    sin_Omega = math.sin(Omega)
    
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    
    cos_omega = math.cos(omega)
    sin_omega = math.sin(omega)
    
    # Calculate semi-latus rectum
    p = a * (1 - e**2)
    # Calculate radius
    r = p / (1 + e * cos_nu)
    
    # Direction cosine matrices
    CN2I = np.array([[cos_Omega, -sin_Omega, 0],
                     [sin_Omega,  cos_Omega, 0],
                     [        0,          0, 1]]) # Longitude of ascending node
    
    CQ2N = np.array([[1,     0,      0],
                     [0, cos_i, -sin_i],
                     [0, sin_i,  cos_i]])         # Orbital inclination
    
    CP2Q = np.array([[cos_omega, -sin_omega, 0],
                     [sin_omega,  cos_omega, 0],
                     [        0,          0, 1]]) # Argument of periapsis
    
    CP2I = np.dot(np.dot(CN2I, CQ2N), CP2Q)
    
    # Calculate r and v in perifocal basis {P_x,P_y,P_z}
    r_P = np.array([r * cos_nu, r * sin_nu, 0])
    v_P = math.sqrt(mu / p) * np.array([-sin_nu, e + cos_nu, 0])
    
    # Convert perifocal basis {P_x,P_y,P_z} to PCI coordinates
    rPCI = np.dot(CP2I, r_P)
    vPCI = np.dot(CP2I, v_P)
    
    return rPCI, vPCI


if __name__ == '__main__':
    # Testing
    import time
    #import cProfile
    oe = np.array([7000, 0.2, 0, 0.8, 0, 1.57])
    mu = 398600
    oper = 1000
    t0 = time.perf_counter()
    for _ in range(oper):
        rPCI, vPCI = oe2rv(oe, mu)
    t1 = time.perf_counter()
    assert np.allclose(rPCI, np.array([5.35046335, 4681.12206183, 4819.86376512]))
    assert np.allclose(vPCI, np.array([-7.70165168, 1.07743174, 1.10936527]))
    print(f'Completed in {(t1-t0)*1e3:.5f} ms')
    print(f'{oper/(t1-t0):.0f} operations/second')
    #cProfile.run('oe2rv(oe, mu)')