import numpy as np
import matplotlib.pyplot as plt

def mercatorDisplay(lonE: np.ndarray[float], lat: np.ndarray[float], color: str = 'w', markerStyle: str = 'P'):
    """
    Plot Earth-relative longitude and geocentric latitude on Earth image to generate groundtrack.
    
    This function takes the Earth-relative longitude and geocentric latitude and plots it on an image of the
    Earth to produce the groundtrack.

    Args
    ----
    
    `lonE` (numpy.ndarray): Vector containing the values of the Earth-relative longitude [rad]
    
    `lat` (numpy.ndarray): Vector containing the values of the geocentric latitude [rad]
    
    `color` (str, optional): Color of the markers. Default is white 'w'
    
    `markerStyle` (str, optional): Style of the markers. Default is a plus 'P'
    
    Returns
    -------
    An image depicting the generated groundtrack using the input latitude and longitude
    """
    
    # Import Earth image
    earth = plt.imread('earth.jpg')
    
    # Plot the groundtrack
    _, ax = plt.subplots()
    ax.imshow(earth, extent=[-180, 180, -90, 90])
    ax.plot(np.degrees(lonE), np.degrees(lat), color + markerStyle)
    ax.set_aspect('equal')
    plt.xticks(np.arange(-180, 181, step=30))
    plt.yticks(np.arange(-90, 91, step=30))
    plt.show()