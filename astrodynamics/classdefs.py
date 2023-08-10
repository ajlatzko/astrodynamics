import numpy as np
import math
from typing import Type
from . import Kepler as pk
from . import quaternion as qtn

# Set constants
G = 6.674e-11
twoPi = 2 * math.pi
halfPi = 0.5 * math.pi
e3 = np.array([0, 0, 1])

class CelestialBody:
    """
    Class to represent a celestial body.

    Attributes
    ----------
    `name` (str): Name of the celestial body
    
    `mass` (float): Mass of the celestial body [kg]
    
    `radius` (float): Radius of the celestial body [m]
    
    `siderealRotation` (float): Sidereal rotation period of the celestial body [s]
    
    `orbit` (astrodynamics.Orbit): Orbit class representing the celestial body's orbit (if applicable)
    
    `atmPressure` (float): Atmospheric pressure at the surface of the celestial body (if applicable) [Pa]
    
    `atmScaleHeight` (float): Scale height of the atmosphere [m]
    
    `gravParameter` (float): Gravitational parameter of the celestial body [m^3/s^2]
    
    `SOI` (float): Sphere of influence of the celestial body [m]
    
    `atmRadius` (float): Effective radius of the atmosphere [m]
    """
    
    
    def __init__(self, name: str, mass: float, radius: float, siderealRotation: float, orbit: Type['Orbit'], atmPressure: float = None, atmScaleHeight: float = None):
        """
        Constructor for the CelestialBody class.

        Parameters
        ----------
        `mass` (float): Mass of the celestial body [kg]
        
        `radius` (float): Radius of the celestial body [m]
        
        `siderealRotation` (float): Sidereal rotation period of the celestial body [s]
        
        `orbit` (astrodynamics.Orbit): Orbit class representing the celestial body's orbit (if applicable)
        
        `atmPressure` (float, optional): Atmospheric pressure at the surface of the celestial body (if applicable) [Pa]
            Defaults to None
        
        `atmScaleHeight` (float, optional): Scale height of the atmosphere [m]
            Defaults to None
        """
        
        # Input validation
        if not isinstance(name, str):
            raise TypeError('Name must be a string')
        if orbit is not None and not isinstance(orbit, Orbit):
            raise TypeError('Orbit must be a valid Orbit object or None')
        if mass <= 0:
            raise ValueError('Mass must be positive')
        if radius <= 0:
            raise ValueError('Radius must be positive')
        
        # Set the properties of the celestial body
        self.name = name
        self.mass = float(mass)
        self.radius = float(radius)
        self.siderealRotation = float(siderealRotation)
        self.orbit = orbit
        self.atmPressure = float(atmPressure) if atmPressure is not None else atmPressure
        self.atmScaleHeight = float(atmScaleHeight) if atmScaleHeight is not None else atmScaleHeight
        
        # Calculate the gravitational parameter of the body
        self.gravParameter = G * self.mass
        
        # Calculate the sphere of influence if the celestial body is in an orbit
        if self.orbit is not None:
            self.SOI = self.orbit.semiMajorAxis * (self.mass / self.orbit.refBody.mass)**0.4
        else:
            self.SOI = None
        
        # Calculate the atmosphere radius if scale height is provided
        if self.atmScaleHeight is not None:
            self.atmRadius = -math.log(1e-6) * self.atmScaleHeight + self.radius
    
    
    def circOrbitVel(self, altitude: float) -> float:
        """
        Calculates the velocity of a circular orbit at a given altitude.

        This function calculates the velocity of a circular orbit at the specified
        altitude above the celestial body's surface.

        Args
        ----
        `altitude` (float): Altitude above the surface of the celestial body [m]

        Returns
        -------
        float: The velocity of a circular orbit at the given altitude [m/s]
        """
        
        return math.sqrt(self.gravParameter / (altitude + self.radius))
    
    
    def escapeVel(self, altitude: float) -> float:
        """
        Calculates the escape velocity at a given altitude.
        
        This function calculates the velocity needed for an object to escape the
        gravitational influence of the celestial body at a specified altitude above
        its surface.
        
        Args
        ----
        `altitude` (float): Altitude above the surface of the celestial body [m]
        
        Returns
        -------
        float: The velocity needed to escape the gravitational influence of the celestial body at the specified altitude [m/s]
        """

        return math.sqrt((2 * self.gravParameter) / (altitude + self.radius))
    
    
    def siderealTimeAt(self, longitude: float, time: float) -> float:
        """
        Calculates the sidereal time at a specific longitude and time.

        This function calculates the sidereal time at the given longitude
        and time on the celestial body.

        Args
        ----
        `longitude` (float): Longitude of the observer [rad]
        
        `time` (float): Time since a reference point [s]

        Returns
        -------
        float: Sidereal time at the given longitude and time [s]
        """
        
        result = ((time / self.siderealRotation) * twoPi + halfPi + longitude) % twoPi
        if result < 0:
            return result + twoPi
        else:
            return result


class Orbit:
    """
    Class to represent the orbital properties of an object around a reference celestial body.
    
    Attributes
    ----------
    `refBody` (astrodynamics.CelestialBody): Reference celestial body for the orbit
    
    `semiMajorAxis` (float): Semi-major axis of the orbit [m]
    
    `eccentricity` (float): Eccentricity of the orbit
    
    `inclination` (float): Inclination of the orbit [rad]
    
    `longOfAscNode` (float): Longitude of the ascending node [rad]
    
    `argOfPeriapsis` (float): Argument of periapsis [rad]
    
    `meanAnomalyEpoch` (float): Mean anomaly at epoch [rad]
    
    `eccAnomalyEpoch` (float): Eccentric anomaly at epoch [rad]
    
    `trueAnomalyEpoch` (float): True anomaly at epoch [rad]
    
    `isBounded` (bool): Flag indicating if the orbit is bounded
    
    `apoapsis` (float): Apoapsis distance [m]
    
    `periapsis` (float): Periapsis distance [m]
    
    `apoapsisAlt` (float): Apoapsis altitude [m]
    
    `periapsisAlt` (float): Periapsis altitude [m]
    
    `semiMinorAxis` (float): Semi-minor axis of the orbit [m]
    
    `semiLatusRectum` (float): Semi-latus rectum of the orbit [m]
    
    `meanMotion` (float): Mean motion of the orbit [rad/s]
    
    `period` (float): Orbital period [s]
    
    `rotToRefFrame` (numpy.ndarray): Quaternion representing rotation to reference frame
    
    `normalVector` (numpy.ndarray): Normal vector to the orbit
    """
    
    
    def __init__(self, refBody: Type['CelestialBody'], semiMajorAxis: float, eccentricity: float, inclination: float, longOfAscNode: float, argOfPeriapsis: float, meanAnomaly: float):
        """
        Constructor for the Orbit class.

        Parameters
        ----------
        `refBody` (astrodynamics.CelestialBody): Reference celestial body for the orbit
        
        `semiMajorAxis` (float): Semi-major axis of the orbit [m]
        
        `eccentricity` (float): Eccentricity of the orbit
        
        `inclination` (float): Inclination of the orbit [deg]
        
        `longOfAscNode` (float): Longitude of the ascending node [deg]
        
        `argOfPeriapsis` (float): Argument of periapsis [deg]
        
        `meanAnomaly` (float): Mean anomaly at epoch [rad]
        """
        
        # Input validation
        if not isinstance(refBody, CelestialBody):
            raise TypeError('Reference body must be a valid CelestialBody object')
        if semiMajorAxis <= 0:
            raise ValueError('Semi-major axis must be positive')
        if eccentricity < 0:
            raise ValueError('Eccentricity cannot be negative')
        
        # Set the properties of the orbit
        self.refBody = refBody
        self.semiMajorAxis = float(semiMajorAxis)
        self.eccentricity = float(eccentricity)
        self.inclination = math.radians(inclination)
        self.longOfAscNode = math.radians(longOfAscNode) % twoPi
        self.argOfPeriapsis = math.radians(argOfPeriapsis) % twoPi
        
        # Compute and normalize initial anomalies
        self.meanAnomalyEpoch = float(meanAnomaly) % twoPi
        self.eccAnomalyEpoch = pk.KeplerSolver(self.eccentricity, self.meanAnomalyEpoch)
        self.trueAnomalyEpoch = 2 * math.atan2(math.sqrt(1 + self.eccentricity) * math.sin(self.eccAnomalyEpoch / 2), math.sqrt(1 - self.eccentricity) * math.cos(self.eccAnomalyEpoch / 2))
        
        # Determine if the orbit is bounded
        if self.eccentricity >= 1:
            self.isBounded = True
        else:
            self.isBounded = False
        
        # Compute additional orbit parameters
        self.apoapsis = self.semiMajorAxis * (1 + self.eccentricity)
        self.periapsis = self.semiMajorAxis * (1 - self.eccentricity)
        self.apoapsisAlt = self.apoapsis - self.refBody.radius
        self.periapsisAlt = self.periapsis - self.refBody.radius
        
        self.semiMinorAxis = self.semiMajorAxis * math.sqrt(1 - self.eccentricity**2)
        self.semiLatusRectum = self.semiMajorAxis * (1 - self.eccentricity**2)
        
        self.meanMotion = math.sqrt(self.refBody.gravParameter / math.fabs(self.semiMajorAxis)**3)
        
        # Compute the orbital period
        if self.isBounded:
            self.period = math.inf
        else:
            self.period = twoPi / self.meanMotion
        
        # Compute the rotation to reference frame quaternion
        axOfInclination = np.array([math.cos(-self.argOfPeriapsis), math.sin(-self.argOfPeriapsis), 0])
        self.rotToRefFrame = qtn.qConcat(qtn.qFromAngleAxis(self.longOfAscNode + self.argOfPeriapsis, e3), qtn.qFromAngleAxis(self.inclination, axOfInclination))
        
        # Compute the normal vector to the orbit
        self.normalVector = qtn.qRotate(self.rotToRefFrame, e3)
    
    
    def radiusAtTrueAnomaly(self, tA: float) -> float:
        """
        Calculates the radius at the given true anomaly.

        This function computes the radius of the orbit at the specified true anomaly.

        Args
        ----
        `tA` (float): True anomaly [rad]

        Returns
        -------
        float: Radius at the given true anomaly [m]
        """
        
        e = self.eccentricity
        return self.semiMajorAxis * (1 - e**2) / (1 + e * math.cos(tA))
    
    
    def altAtTrueAnomaly(self, tA: float) -> float:
        """
        Calculates the altitude at the given true anomaly.

        This function computes the altitude of the object at the specified true anomaly.

        Args
        ----
        `tA` (float): True anomaly [rad]

        Returns
        -------
        float: Altitude at the given true anomaly [m]
        """
        
        return self.radiusAtTrueAnomaly(tA) - self.refBody.radius
    
    
    def speedAtTrueAnomaly(self, tA: float) -> float:
        """
        Calculates the speed at the given true anomaly.

        This function computes the speed of the object at the specified true anomaly.

        Args
        ----
        `tA` (float): True anomaly [rad]

        Returns
        -------
        float: Speed at the given true anomaly [m/s]
        """
        
        return math.sqrt(self.refBody.gravParameter * (2 / self.radiusAtTrueAnomaly(tA) - 1 / self.semiMajorAxis))
    
    
    def posAtTrueAnomaly(self, tA: float) -> np.ndarray[float]:
        """
        Calculates the position vector at the given true anomaly.

        This function computes the position vector of the object at the specified
        true anomaly in the reference frame.

        Args
        ----
        `tA` (float): True anomaly [rad]

        Returns
        -------
        numpy.ndarray: Position vector at the given true anomaly [m]
        """
        
        r = self.radiusAtTrueAnomaly(tA)
        return qtn.qRotate(self.rotToRefFrame, np.array([r*math.cos(tA), r*math.sin(tA), 0]))
    
    
    def velocityAtTrueAnomaly(self, tA: float) -> np.ndarray:
        """
        Calculates the velocity vector at the given true anomaly.

        This function computes the velocity vector of the object at the specified
        true anomaly in the reference frame.

        Args
        ----
        `tA` (float): True anomaly [rad]

        Returns
        -------
        numpy.ndarray: Velocity vector at the given true anomaly [m/s]
        """
        
        mu = self.refBody.gravParameter
        e = self.eccentricity
        h = math.sqrt(mu * self.semiMajorAxis * (1 - e**2))
        r = self.radiusAtTrueAnomaly(tA)
        s = math.sin(tA)
        c = math.cos(tA)
        vr = mu * e * s / h
        vtA = h / r
        return qtn.qRotate(self.rotToRefFrame, np.array([vr*c - vtA*s, vr*s + vtA*c, 0]))
    
    
    def trueAnomalyAtPos(self, p: np.ndarray[float]) -> float:
        """
        Calculates the true anomaly at the given position vector.

        This function computes the true anomaly of the object at the specified
        position vector in the reference frame.

        Args
        ----
        `p` (numpy.ndarray): Position vector [m]

        Returns
        -------
        float: True anomaly at the given position [rad]
        """
        
        p = qtn.qRotate(qtn.qConj(self.rotToRefFrame), p)
        return math.atan2(p[1], p[0])
    
    
    def meanAnomalyAt(self, t: float) -> float:
        """
        Calculates the mean anomaly at the given time.

        This function computes the mean anomaly of the object at the specified time.

        Args
        ----
        `t` (float): Time [s]

        Returns
        -------
        float: Mean anomaly at the given time [rad]
        """
        
        return self.meanAnomalyEpoch + self.meanMotion * (t % self.period) % twoPi
    
    
    def eccAnomalyAt(self, t: float) -> float:
        """
        Calculates the eccentric anomaly at the given time.

        This function computes the eccentric anomaly of the object at the specified time.

        Args
        ----
        `t` (float): Time [s]

        Returns
        -------
        float: Eccentric anomaly at the given time [rad]
        """
        
        M = self.meanAnomalyAt(t)
        E = pk.KeplerSolver(self.eccentricity, M)
        if E < 0:
            return E + twoPi
        elif E > twoPi:
            return E - twoPi
        else:
            return E
    
    
    def trueAnomalyAt(self, t: float) -> float:
        """
        Calculates the true anomaly at the given time.

        This function computes the true anomaly of the object at the specified time.

        Args
        ----
        `t` (float): Time [s]

        Returns
        -------
        float: True anomaly at the given time [rad]
        """
        
        E = self.eccAnomalyAt(t)
        tA = 2 * math.atan2(math.sqrt(1 + self.eccentricity) * math.sin(E / 2), math.sqrt(1 - self.eccentricity) * math.cos(E / 2))
        if tA < 0:
            return tA + twoPi
        else:
            return tA
    
    
    def eccAnomalyAtTrueAnomaly(self, tA: float) -> float:
        """
        Calculates the eccentric anomaly at the given true anomaly.

        This function computes the eccentric anomaly of the object at the specified true anomaly.

        Args
        ----
        `tA` (float): True anomaly [rad]

        Returns
        -------
        float: Eccentric anomaly at the given true anomaly [rad]
        """
        
        E = 2 * math.atan(math.tan(tA / 2) / math.sqrt((1 + self.eccentricity) / (1 - self.eccentricity)))
        if E < 0:
            return E + twoPi
        else:
            return E
    
    
    def meanAnomalyAtTrueAnomaly(self, tA: float) -> float:
        """
        Calculates the mean anomaly at the given true anomaly.

        This function computes the mean anomaly of the object at the specified true anomaly.

        Args
        ----
        `tA` (float): True anomaly [rad]

        Returns
        -------
        float: Mean anomaly at the given true anomaly [rad]
        """
        
        E = self.eccAnomalyAtTrueAnomaly(tA)
        return E - self.eccentricity * math.sin(E)
    
    
    def timeAtTrueAnomaly(self, tA: float, t0: float = 0) -> float:
        """
        Calculates the time at the given true anomaly.

        This function computes the time at the specified true anomaly. If an initial time
        `t0` is given, this function computes the time at the specified true anomaly with
        initial time `t0`.

        Args
        ----
        `tA` (float): True anomaly [rad]
        
        `t0` (float, optional): Initial time [s]
            Defaults to 0

        Returns
        -------
        float: Time at the specified true anomaly [s]
        """
        
        M = self.meanAnomalyAtTrueAnomaly(tA)
        t = (t0 - (t0 % self.period)) + (M - self.meanAnomalyEpoch) / self.meanMotion
        if t < t0:
            return t + self.period
        else:
            return t
    
    
    def phaseAngle(self, orbit, t: float) -> float:
        """
        Calculates the phase angle between two orbits at the given time.

        This function computes the phase angle between the current orbit and the
        specified orbit at the given time.

        Args
        ----
        `orbit` (astrodynamics.Orbit): Specified orbit
        
        `t` (float): Time [s]

        Returns
        -------
        float: Phase angle at time `t` [rad]
        """
        
        n = self.normalVector
        p1 = self.posAtTrueAnomaly(self.trueAnomalyAt(t))
        p2 = orbit.posAtTrueAnomaly(orbit.trueAnomalyAt(t))
        p2 = p2 - n * np.dot(p2, n)
        r1 = np.linalg.norm(p1)
        r2 = np.linalg.norm(p2)
        angle = math.arccos(np.dot(p1, p2) / (r1 * r2))
        if np.dot(np.cross(p1, p2), n) < 0:
            angle = twoPi - angle
        if orbit.semiMajorAxis < self.semiMajorAxis:
            angle = angle - twoPi
        return angle


if __name__ == '__main__':
    # Testing
    import time
    Kerbol = CelestialBody('Kerbol', 1.756567e28, 2.616e8, 432000, None)
    Kerbin = CelestialBody('Kerbin', 5.2915793e22, 600000, 21600, Orbit(Kerbol, 13599840256, 0.0, 0, 0, 0, 3.14), 1, 5066.77)
    Duna = CelestialBody('Duna', 4.5154812e21, 320000, 65517.859, Orbit(Kerbol, 20726155264, 0.051, 0.06, 135.5, 0, 3.14), 0.2, 3619.12)
    
    resolution = 100

    UTdepRange = (0, 18407090)
    flightTimeRange = (3261600, 9784800)

    UTdeps = np.linspace(UTdepRange[0], UTdepRange[1], resolution)
    fltTimes = np.linspace(flightTimeRange[0], flightTimeRange[1], resolution)
    UTarrs = np.zeros((resolution,resolution))
    for i in range(resolution):
        UTarrs[i,:] = UTdeps + fltTimes[i]

    # Generate r and v vectors for each ut
    rArrayDep = np.zeros((resolution, 3))
    vArrayDep = np.zeros((resolution, 3))
    rArrayArr = np.zeros((resolution, 3, resolution))
    vArrayArr = np.zeros((resolution, 3, resolution))
    t0 = time.perf_counter()
    for i in range(resolution):
        rArrayDep[i,:] = Kerbin.orbit.posAtTrueAnomaly(Kerbin.orbit.trueAnomalyAt(UTdeps[i]))
        vArrayDep[i,:] = Kerbin.orbit.velocityAtTrueAnomaly(Kerbin.orbit.trueAnomalyAt(UTdeps[i]))
        for j in range(resolution):
            rArrayArr[i,:,j] = Duna.orbit.posAtTrueAnomaly(Duna.orbit.trueAnomalyAt(UTarrs[j,i]))
            vArrayArr[i,:,j] = Duna.orbit.velocityAtTrueAnomaly(Duna.orbit.trueAnomalyAt(UTarrs[j,i]))
    
    t1 = time.perf_counter()
    print(f'Completed in {(t1-t0)*1e3:0.5f} ms')