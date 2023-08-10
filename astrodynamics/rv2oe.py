import numpy as np
import math

def rv2oe(rPCI: np.ndarray[float], vPCI: np.ndarray[float], mu: float) -> np.ndarray[float]:
    """
    Convert planet-centered inertial position and velocity vectors to orbital elements.

    Given the position and velocity vectors in the planet-centered inertial (PCI) coordinate system,
    this function computes the six classical orbital elements representing the orbit of a satellite.

    Args
    ----
    `rPCI` (numpy.ndarray): The position vector in the planet-centered inertial frame
    
    `vPCI` (numpy.ndarray): The velocity vector in the planet-centered inertial frame
    
    `mu` (float): µ, gravitational parameter of the central body

    Returns
    -------
    `oe` (numpy.ndarray): Six classical orbital elements in the following order:
        `oe[0]`: Semi-major axis (a)
        
        `oe[1]`: Eccentricity (e)
        
        `oe[2]`: Longitude of the ascending node (Ω, Omega) [rad]
        
        `oe[3]`: Orbital inclination (i) [rad]
        
        `oe[4]`: Argument of periapsis (ω, omega) [rad]
        
        `oe[5]`: True anomaly (ν, nu) [rad]
        
    Example
    -------
    >>> import numpy as np
    >>> rPCI = np.array([0, 5000, 4500])
    >>> vPCI = np.array([-7.7, 1, 1])
    >>> mu = 398600
    >>> oe = rv2oe(rPCI, vPCI, mu)
    >>> print(oe)
    [6.96599610e+03 1.83527109e-01 1.44290130e-02 7.32866865e-01 6.27614927e+00 1.56710753e+00]
    """
    
    # Input validation
    if not isinstance(rPCI, np.ndarray) or not isinstance(vPCI, np.ndarray):
        raise TypeError('Input vectors must be provided as a numpy.ndarray')
    if rPCI.shape != (3,) or vPCI.shape != (3,):
        raise ValueError('Input vectors must be a 3-element numpy.ndarray: shape is not (3,)')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    # Create required basis vectors
    Iz = np.array([0, 0, 1])
    
    # Intermediate calculations
    h_vec = np.cross(rPCI, vPCI)                                     # Calculate specific angular momentum vector
    h = np.linalg.norm(h_vec)                                        # Normalize specific angular momentum vector
    n = np.cross(Iz, h_vec)                                          # Calculate line of nodes vector
    e_vec = np.cross(vPCI, h_vec) / mu - rPCI / np.linalg.norm(rPCI) # Calculate eccentricity vector
    e = np.linalg.norm(e_vec)                                        # Normalize eccentricity vector
    p = h**2 / mu                                                    # Calculate semi-latus rectum

    a = p / (1 - e**2)
    Omega = math.atan2(n[1], n[0])                                                 # Calculate longitude of ascending node
    i = math.atan2(np.dot(h_vec, np.cross(n, Iz)), np.linalg.norm(n) * h_vec[2])   # Calculate orbital inclination
    omega = math.atan2(np.dot(e_vec, np.cross(h_vec, n)), h * np.dot(e_vec, n))    # Calculate argument of periapsis
    nu = math.atan2(np.dot(rPCI, np.cross(h_vec, e_vec)), h * np.dot(rPCI, e_vec)) # Calculate true anomaly
    
    # Check if the angles are negative and add 2*pi if they are
    twoPi = 2 * math.pi
    if Omega < 0:
        Omega += twoPi
    if i < 0:
        i += twoPi
    if omega < 0:
        omega += twoPi
    if nu < 0:
        nu += twoPi

    return np.array([a, e, Omega, i, omega, nu])


if __name__ == '__main__':
    # Testing
    import time
    #import cProfile
    rPCI = np.array([0, 5000, 4500])
    vPCI = np.array([-7.7, 1, 1])
    mu = 398600
    oper = 1000
    t0 = time.perf_counter()
    for _ in range(oper):
        oe = rv2oe(rPCI, vPCI, mu)
    t1 = time.perf_counter()
    assert np.allclose(oe, np.array([6.96599610e+03, 1.83527109e-01, 1.44290130e-02, 7.32866865e-01,  6.27614927e+00, 1.56710753e+00]))
    print(f'Completed in {(t1-t0)*1e3:0.5f} ms')
    print(f'{oper/(t1-t0):.0f} operations/second')
    #cProfile.run('rv2oe(rPCI, vPCI, mu)')