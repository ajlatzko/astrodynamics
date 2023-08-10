import numpy as np
import math

def propagateOnCircle(rt0: np.ndarray[float], vt0: np.ndarray[float], t0: float, tf: float, mu: float, N: int) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Propagates a circular orbit around a celestial body.
    
    This function computes the PCI position and PCI inertial velocity
    for `N` time intervals between `t0` and `tf` for a circular orbit.

    Args
    ----
    `rt0` (numpy.ndarray): Initial position vector in the planet-centered inertial frame
    
    `vt0` (numpy.ndarray): Initial velocity vector in the planet-centered inertial frame
    
    `t0` (float): Initial time
    
    `tf` (float): Final time
    
    `mu` (float): µ, gravitational parameter of the central body
    
    `N` (int): Number of time intervals

    Returns
    -------
    tuple: A tuple containing three objects:
        `tspan` (numpy.ndarray): An array of size (N,) containing time values from t0 to tf with N intervals
        
        `rt` (numpy.ndarray): An array of size (N,3) containing position vectors in the planet-centered inertial frame
        
        `vt` (numpy.ndarray): An array of size (N,3) containing velocity vectors in the planet-centered inertial frame
    
    Example
    -------
    >>> import numpy as np
    >>> rt0 = np.array([-5613.97603835865, -2446.44383433555, 2600.48533877841])
    >>> vt0 = np.array([2.12764777374332, -7.13421216656605, -2.1184067703542])
    >>> t0 = 0
    >>> tf = 600
    >>> mu = 398600
    >>> N = 3
    >>> tspan, rt, vt = propagateOnCircle(rt0, vt0, t0, tf, mu, N)
    >>> print(tspan)
    [  0. 300. 600.]
    >>> print(rt)
    [[-5613.97603836 -2446.44383434  2600.48533878]
     [-4650.08714957 -4396.01921373  1821.00053951]
     [-3125.47724472 -5815.50982679   821.93420554]]
    >>> print(vt)
    [[ 2.12764777 -7.13421217 -2.11840677]
     [ 4.23291444 -5.73075146 -3.02529975]
     [ 5.82776399 -3.63626009 -3.5673933 ]]
    """
    
    # Input validation
    if not isinstance(rt0, np.ndarray) or not isinstance(vt0, np.ndarray):
        raise TypeError('Input vectors must be provided as a numpy.ndarray')
    if rt0.shape != (3,) or vt0.shape != (3,):
        raise ValueError('Input vectors must be a 3-element numpy.ndarray: shape is not (3,)')
    if not isinstance(N, int):
        raise TypeError('Number of time intervals must be an integer')
    if N <= 0:
        raise ValueError('Number of time intervals must be positive')
    if tf <= t0:
        raise ValueError('Final time must be greater than initial time')
    if mu <= 0:
        raise ValueError('The gravitational parameter must be positive')
    
    # Create required basis vectors
    Iz = np.array([0, 0, 1])
    
    # Initialize vectors
    rt_w = np.zeros((N,3))
    vt_w = np.zeros((N,3))
    rt = np.zeros((N,3))
    vt = np.zeros((N,3))
    
    # Intermediate calculations
    h = np.cross(rt0, vt0)                                                          # Calculate specific angular momentum vector
    n = np.cross(Iz, h)                                                             # Calculate line of nodes vector
    Omega = math.atan2(n[1], n[0])                                                  # Calculate longitude of ascending node
    i = math.atan2(np.dot(h, np.cross(n, Iz)), np.linalg.norm(n) * h[2])            # Calculate orbital inclination
    u = math.atan2(np.dot(rt0, np.cross(h, n)), np.linalg.norm(h) * np.dot(rt0, n)) # Calculate argument of latitude (circular orbit)
    
    cos_Omega = math.cos(Omega)
    sin_Omega = math.sin(Omega)
    
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    
    cos_u = math.cos(u)
    sin_u = math.sin(u)
    
    # Direction cosine matrices
    CN2I = np.array([[cos_Omega, -sin_Omega, 0],
                     [sin_Omega,  cos_Omega, 0],
                     [        0,          0, 1]]) # Longitude of ascending node
    
    CQ2N = np.array([[1,     0,      0],
                     [0, cos_i, -sin_i],
                     [0, sin_i,  cos_i]])         # Orbital inclination
    
    CP2Q = np.array([[cos_u, -sin_u, 0],
                     [sin_u,  cos_u, 0],
                     [    0,      0, 1]])         # Argument of latitude
    
    CP2I = np.dot(np.dot(CN2I, CQ2N), CP2Q)
    
    tspan = np.linspace(t0, tf, N)   # Compute times for N intervals on [t0,tf]
    a = np.linalg.norm(rt0)          # Compute semi-major axis of circular orbit
    thetadot = math.sqrt(mu / a**3)  # Compute angular velocity (constant for circular orbit)
    
    # Run vectorized code for large values of N
    if N >= 6:
        # Calculate array of thetas and sines and cosines
        thetas = thetadot * (tspan - t0)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        
        # Compute r and v in basis {w1,w2,w3}
        rt_w = np.column_stack((a * cos_thetas, a * sin_thetas, np.zeros(N)))
        vt_w = np.column_stack((-a * thetadot * sin_thetas, a * thetadot * cos_thetas, np.zeros(N)))
        
        # Transform {w1,w2,w3} to ECI coordinates
        rt = np.dot(rt_w, CP2I.T)
        vt = np.dot(vt_w, CP2I.T)
    else:
        for i in range(N):
            # Compute theta value for time t and sine and cosine
            theta = thetadot * (tspan[i] - t0)
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            
            # Compute r and v in basis {w1,w2,w3}
            rt_w[i,:] = np.array([a * cos_theta, a * sin_theta, 0])
            vt_w[i,:] = np.array([-a * thetadot * sin_theta, a * thetadot * cos_theta, 0])

            # Transform {w1,w2,w3} to ECI coordinates
            rt[i,:] = np.dot(CP2I, rt_w[i,:])
            vt[i,:] = np.dot(CP2I, vt_w[i,:])
        
    return tspan, rt, vt


if __name__ == '__main__':
    # Testing
    import time
    #import cProfile
    rt0 = np.array([-5613.97603835865, -2446.44383433555, 2600.48533877841])
    vt0 = np.array([2.12764777374332, -7.13421216656605, -2.1184067703542])
    t0 = 0
    tf = 600
    mu = 398600
    N = 10000
    to = time.perf_counter()
    tspan, rt, vt = propagateOnCircle(rt0, vt0, t0, tf, mu, N)
    t1 = time.perf_counter()
    print(f'Completed in {(t1-to)*1e3:0.5f} ms')
    #cProfile.run('propagateOnCircle(rt0, vt0, t0, tf, mu, N)')