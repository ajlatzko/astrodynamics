import numpy as np

def earthSphere(n: int = 50) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Generate an Earth-sized sphere.
    
    This function computes the x, y, and z values so that plotting a surface with
    these values produces a sphere with radius equal to the radius of Earth in km.

    Args
    ----
    `n` (int, optional): Resolution of sphere. Default is 50
        Must be greater than 10

    Returns
    -------
    tuple: A tuple containing three objects:
        `x` (numpy.ndarray): (n+1, n+1) matrix with x values of the sphere
        
        `y` (numpy.ndarray): (n+1, n+1) matrix with y values of the sphere
        
        `z` (numpy.ndarray): (n+1, n+1) matrix with z values of the sphere
    
    Note
    ----
    Use `color=(0, 0.176, 1)` for the color of Earth.
    
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> x, y, z = earthSphere()
    >>> fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    >>> ax.plot_surface(x, y, z, color=(0, 0.176, 1))
    >>> ax.axis('equal')
    >>> plt.show()
    """
    
    if not isinstance(n, int):
        raise TypeError('Resolution must be an integer')
    if n < 10:
        raise ValueError('Resolution must be greater than or equal to 10')
    
    scale = 6378.145
    
    # -pi <= theta <= pi is a row vector
    # -pi/2 <= phi <= pi/2 is a column vector
    theta = np.arange(-n, n+1, 2) / n * np.pi
    theta = np.reshape(theta, (1, n+1))
    phi = np.arange(-n, n+1, 2) / n * np.pi / 2
    phi = np.reshape(phi, (n+1, 1))
    
    cosphi = np.cos(phi)
    cosphi[0,0] = 0
    cosphi[-1,0] = 0
    sintheta = np.sin(theta)
    sintheta[0,0] = 0
    sintheta[0,-1] = 0
    
    x = scale * cosphi @ np.cos(theta)
    y = scale * cosphi @ sintheta
    z = scale * np.sin(phi) @ np.ones((1,n+1))
    
    return x, y, z


if __name__ == '__main__':
    # Testing
    import time
    import matplotlib.pyplot as plt
    #import cProfile
    t0 = time.perf_counter()
    x, y, z = earthSphere()
    t1 = time.perf_counter()
    print(f'Completed in {(t1-t0)*1e3:0.5f} ms')
    #cProfile.run('earthSphere()')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, y, z, color=(0, 0.176, 1))    
    ax.axis('equal')
    plt.show()