"""Orbit states, Keplerian elements, and the conversions between them.

Pure functions over immutable values. Nothing here performs input or output and
nothing here advances time; propagation lives in the algorithm layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from conjunction_screening.model.arrays import Matrix, Vector, as_vector
from conjunction_screening.model.constants import MU_EARTH

__all__ = [
    "KeplerianElements",
    "OrbitState",
    "eccentric_to_mean_anomaly",
    "eccentric_to_true_anomaly",
    "element_state_jacobian",
    "elements_from_state",
    "path_positions",
    "perifocal_to_inertial",
    "solve_kepler_equation",
    "state_from_elements",
    "state_from_mean_elements",
    "true_to_eccentric_anomaly",
]

_CIRCULAR_TOLERANCE: Final[float] = 1e-11
"""Eccentricity below which the eccentricity vector carries no reliable direction."""

_EQUATORIAL_TOLERANCE: Final[float] = 1e-11
"""Relative node-vector magnitude below which the ascending node is undefined."""

_TWO_PI: Final[float] = 2.0 * np.pi


@dataclass(frozen=True, slots=True)
class OrbitState:
    """An inertial Cartesian state at a single epoch.

    Attributes:
        epoch_s: Seconds from the screening epoch.
        position_m: Inertial position in m.
        velocity_m_s: Inertial velocity in m/s.
    """

    epoch_s: float
    position_m: Vector
    velocity_m_s: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch_s", float(self.epoch_s))
        object.__setattr__(self, "position_m", as_vector(self.position_m, 3, "position_m"))
        object.__setattr__(self, "velocity_m_s", as_vector(self.velocity_m_s, 3, "velocity_m_s"))

    @property
    def radius_m(self) -> float:
        """Distance from the central body centre in m."""
        return float(np.linalg.norm(self.position_m))

    @property
    def speed_m_s(self) -> float:
        """Inertial speed in m/s."""
        return float(np.linalg.norm(self.velocity_m_s))

    @property
    def vector(self) -> Vector:
        """The six-element state vector, position then velocity."""
        return np.concatenate((self.position_m, self.velocity_m_s))

    @classmethod
    def from_vector(cls, vector: Vector, epoch_s: float) -> OrbitState:
        """Build a state from a six-element position and velocity vector."""
        values = as_vector(vector, 6, "vector")
        return cls(epoch_s=epoch_s, position_m=values[:3], velocity_m_s=values[3:])


@dataclass(frozen=True, slots=True)
class KeplerianElements:
    """Classical orbital elements of a closed orbit.

    Attributes:
        semi_major_axis_m: Semi-major axis in m, strictly positive.
        eccentricity: Eccentricity in [0, 1).
        inclination_rad: Inclination in [0, pi].
        raan_rad: Right ascension of the ascending node in [0, 2 pi).
        arg_perigee_rad: Argument of perigee in [0, 2 pi).
        true_anomaly_rad: True anomaly in [0, 2 pi).
        gravitational_parameter: Central body mu in m^3 / s^2.
    """

    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    true_anomaly_rad: float
    gravitational_parameter: float = MU_EARTH

    def __post_init__(self) -> None:
        if not self.semi_major_axis_m > 0.0:
            raise ValueError("semi_major_axis_m must be positive for a closed orbit")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must lie in [0, 1) for a closed orbit")
        if not self.gravitational_parameter > 0.0:
            raise ValueError("gravitational_parameter must be positive")

    @property
    def semi_latus_rectum_m(self) -> float:
        """Semi-latus rectum ``a (1 - e^2)`` in m."""
        return self.semi_major_axis_m * (1.0 - self.eccentricity**2)

    @property
    def perigee_radius_m(self) -> float:
        """Perigee radius ``a (1 - e)`` in m."""
        return self.semi_major_axis_m * (1.0 - self.eccentricity)

    @property
    def apogee_radius_m(self) -> float:
        """Apogee radius ``a (1 + e)`` in m."""
        return self.semi_major_axis_m * (1.0 + self.eccentricity)

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion ``sqrt(mu / a^3)`` in rad/s."""
        return float(np.sqrt(self.gravitational_parameter / self.semi_major_axis_m**3))

    @property
    def period_s(self) -> float:
        """Orbital period in s."""
        return _TWO_PI / self.mean_motion_rad_s

    @property
    def path_speed_bound_m_rad(self) -> float:
        """An upper bound on ``|d r / d nu|`` over the whole orbit, in m/rad.

        With ``r = p / (1 + e cos nu)`` the path derivative magnitude is
        ``r sqrt(1 + (r e sin nu / p)^2)``. Both factors are largest at apogee,
        where ``r = a (1 + e)`` and ``r e / p = e / (1 - e)``. The bound is
        therefore closed form and holds for every true anomaly, which is what
        makes it usable as a Lipschitz constant in the orbit path filter.
        """
        ratio = self.eccentricity / (1.0 - self.eccentricity)
        return self.apogee_radius_m * float(np.sqrt(1.0 + ratio**2))

    def at_true_anomaly(self, true_anomaly_rad: float) -> KeplerianElements:
        """Return a copy of these elements at a different true anomaly."""
        return replace(self, true_anomaly_rad=float(true_anomaly_rad % _TWO_PI))


def true_to_eccentric_anomaly(
    true_anomaly_rad: np.ndarray | float, eccentricity: float
) -> np.ndarray:
    """Convert true anomaly to eccentric anomaly for an elliptical orbit."""
    nu = np.asarray(true_anomaly_rad, dtype=np.float64)
    half = 0.5 * nu
    angle = np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(half),
        np.sqrt(1.0 + eccentricity) * np.cos(half),
    )
    return np.asarray(2.0 * angle, dtype=np.float64)


def eccentric_to_true_anomaly(
    eccentric_anomaly_rad: np.ndarray | float, eccentricity: float
) -> np.ndarray:
    """Convert eccentric anomaly to true anomaly for an elliptical orbit."""
    ecc_anomaly = np.asarray(eccentric_anomaly_rad, dtype=np.float64)
    half = 0.5 * ecc_anomaly
    angle = np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(half),
        np.sqrt(1.0 - eccentricity) * np.cos(half),
    )
    return np.asarray(2.0 * angle, dtype=np.float64)


def eccentric_to_mean_anomaly(
    eccentric_anomaly_rad: np.ndarray | float, eccentricity: float
) -> np.ndarray:
    """Apply Kepler's equation ``M = E - e sin E``."""
    ecc_anomaly = np.asarray(eccentric_anomaly_rad, dtype=np.float64)
    return ecc_anomaly - eccentricity * np.sin(ecc_anomaly)


def solve_kepler_equation(
    mean_anomaly_rad: np.ndarray | float,
    eccentricity: float,
    tolerance: float = 1e-13,
    max_iterations: int = 80,
) -> np.ndarray:
    """Invert Kepler's equation by Newton iteration, vectorised over the input.

    Args:
        mean_anomaly_rad: Mean anomaly, any shape.
        eccentricity: Orbit eccentricity in [0, 1).
        tolerance: Convergence threshold on the Newton step, in rad.
        max_iterations: Iteration cap before the solve is declared failed.

    Returns:
        Eccentric anomaly wrapped to ``(-pi, pi]``.

    Raises:
        RuntimeError: If the iteration does not converge, so that a caller never
            silently consumes a non-converged solve.
    """
    mean = np.asarray(mean_anomaly_rad, dtype=np.float64)
    wrapped = np.mod(mean + np.pi, _TWO_PI) - np.pi
    # Below an eccentricity of 0.8 the mean anomaly is already a good starting
    # point; above it Newton can overshoot, and starting at the nearer apse is
    # the classical remedy.
    ecc_anomaly = wrapped.copy() if eccentricity < 0.8 else np.where(wrapped >= 0.0, np.pi, -np.pi)
    for _ in range(max_iterations):
        residual = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - wrapped
        derivative = 1.0 - eccentricity * np.cos(ecc_anomaly)
        step = residual / derivative
        ecc_anomaly = ecc_anomaly - step
        if float(np.max(np.abs(step))) < tolerance:
            return np.asarray(ecc_anomaly, dtype=np.float64)
    raise RuntimeError(
        f"Kepler equation did not converge in {max_iterations} iterations "
        f"for eccentricity {eccentricity}"
    )


def perifocal_to_inertial(
    raan_rad: float, inclination_rad: float, arg_perigee_rad: float
) -> Matrix:
    """Return the 3 by 3 rotation from the perifocal frame to the inertial frame."""
    cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)
    cos_inc, sin_inc = np.cos(inclination_rad), np.sin(inclination_rad)
    cos_argp, sin_argp = np.cos(arg_perigee_rad), np.sin(arg_perigee_rad)
    return np.array(
        [
            [
                cos_raan * cos_argp - sin_raan * sin_argp * cos_inc,
                -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc,
                sin_raan * sin_inc,
            ],
            [
                sin_raan * cos_argp + cos_raan * sin_argp * cos_inc,
                -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc,
                -cos_raan * sin_inc,
            ],
            [sin_inc * sin_argp, sin_inc * cos_argp, cos_inc],
        ],
        dtype=np.float64,
    )


def state_from_elements(elements: KeplerianElements, epoch_s: float = 0.0) -> OrbitState:
    """Convert classical elements to an inertial Cartesian state."""
    p = elements.semi_latus_rectum_m
    e = elements.eccentricity
    nu = elements.true_anomaly_rad
    mu = elements.gravitational_parameter

    radius = p / (1.0 + e * np.cos(nu))
    position_pf = np.array([radius * np.cos(nu), radius * np.sin(nu), 0.0], dtype=np.float64)
    factor = float(np.sqrt(mu / p))
    velocity_pf = np.array([-factor * np.sin(nu), factor * (e + np.cos(nu)), 0.0], dtype=np.float64)

    rotation = perifocal_to_inertial(
        elements.raan_rad, elements.inclination_rad, elements.arg_perigee_rad
    )
    return OrbitState(
        epoch_s=epoch_s,
        position_m=rotation @ position_pf,
        velocity_m_s=rotation @ velocity_pf,
    )


def path_positions(elements: KeplerianElements, true_anomalies_rad: np.ndarray) -> Matrix:
    """Return inertial positions along the orbit path at the given true anomalies.

    Args:
        elements: Orbit whose geometric path is sampled. The element's own true
            anomaly is ignored; the path is a closed curve independent of phase.
        true_anomalies_rad: Array of true anomalies of any shape ``(..., )``.

    Returns:
        Array of shape ``(..., 3)`` of inertial positions in m.
    """
    nu = np.asarray(true_anomalies_rad, dtype=np.float64)
    radius = elements.semi_latus_rectum_m / (1.0 + elements.eccentricity * np.cos(nu))
    perifocal = np.stack((radius * np.cos(nu), radius * np.sin(nu), np.zeros_like(nu)), axis=-1)
    rotation = perifocal_to_inertial(
        elements.raan_rad, elements.inclination_rad, elements.arg_perigee_rad
    )
    return np.asarray(perifocal @ rotation.T, dtype=np.float64)


def state_from_mean_elements(
    values: np.ndarray, gravitational_parameter: float = MU_EARTH
) -> Vector:
    """Return the six-element state vector for elements ordered by mean anomaly.

    Args:
        values: ``(a, e, i, raan, argp, mean_anomaly)`` with lengths in m and
            angles in rad.
        gravitational_parameter: Central body mu in m^3 / s^2.

    Returns:
        The inertial state vector, position then velocity.
    """
    semi_major, eccentricity, inclination, raan, arg_perigee, mean_anomaly = (
        float(component) for component in np.asarray(values, dtype=np.float64)
    )
    ecc_anomaly = solve_kepler_equation(np.asarray(mean_anomaly, dtype=np.float64), eccentricity)
    true_anomaly = float(eccentric_to_true_anomaly(ecc_anomaly, eccentricity))
    elements = KeplerianElements(
        semi_major_axis_m=semi_major,
        eccentricity=eccentricity,
        inclination_rad=inclination,
        raan_rad=raan % _TWO_PI,
        arg_perigee_rad=arg_perigee % _TWO_PI,
        true_anomaly_rad=true_anomaly % _TWO_PI,
        gravitational_parameter=gravitational_parameter,
    )
    return state_from_elements(elements).vector


def element_state_jacobian(elements: KeplerianElements) -> Matrix:
    """Return ``d(position, velocity) / d(a, e, i, raan, argp, mean anomaly)``.

    Built by central differences of :func:`state_from_mean_elements`. Steps are
    relative for the semi-major axis and absolute for the angles and the
    eccentricity, because both of the latter are legitimately close to zero for
    the near-circular orbits this library screens and a relative step would
    collapse there.
    """
    eccentricity = elements.eccentricity
    mean_anomaly = float(
        eccentric_to_mean_anomaly(
            true_to_eccentric_anomaly(elements.true_anomaly_rad, eccentricity), eccentricity
        )
    )
    base = np.array(
        [
            elements.semi_major_axis_m,
            eccentricity,
            elements.inclination_rad,
            elements.raan_rad,
            elements.arg_perigee_rad,
            mean_anomaly,
        ],
        dtype=np.float64,
    )
    steps = np.array(
        [1e-7 * elements.semi_major_axis_m, 1e-9, 1e-8, 1e-8, 1e-8, 1e-8], dtype=np.float64
    )
    jacobian = np.empty((6, 6), dtype=np.float64)
    for column in range(6):
        perturbation = np.zeros(6, dtype=np.float64)
        perturbation[column] = steps[column]
        forward = state_from_mean_elements(base + perturbation, elements.gravitational_parameter)
        backward = state_from_mean_elements(base - perturbation, elements.gravitational_parameter)
        jacobian[:, column] = (forward - backward) / (2.0 * steps[column])
    return jacobian


def elements_from_state(
    state: OrbitState, gravitational_parameter: float = MU_EARTH
) -> KeplerianElements:
    """Convert an inertial Cartesian state to classical elements.

    Degenerate geometries are resolved by the usual conventions: for a circular
    orbit the argument of perigee is set to zero and the true anomaly becomes the
    argument of latitude; for an equatorial orbit the node is set to zero and the
    argument of perigee becomes the longitude of perigee.
    """
    mu = gravitational_parameter
    r_vec = np.asarray(state.position_m, dtype=np.float64)
    v_vec = np.asarray(state.velocity_m_s, dtype=np.float64)
    radius = float(np.linalg.norm(r_vec))
    speed = float(np.linalg.norm(v_vec))
    if radius == 0.0:
        raise ValueError("position must be non-zero to define an orbit")

    momentum = np.cross(r_vec, v_vec)
    momentum_norm = float(np.linalg.norm(momentum))
    if momentum_norm == 0.0:
        raise ValueError("state is radial and has no orbit plane")

    node = np.cross(np.array([0.0, 0.0, 1.0]), momentum)
    node_norm = float(np.linalg.norm(node))

    radial_rate = float(np.dot(r_vec, v_vec))
    ecc_vec = ((speed**2 - mu / radius) * r_vec - radial_rate * v_vec) / mu
    eccentricity = float(np.linalg.norm(ecc_vec))

    energy = 0.5 * speed**2 - mu / radius
    if energy >= 0.0:
        raise ValueError("state is on an open orbit; this library handles closed orbits only")
    semi_major_axis = -mu / (2.0 * energy)

    inclination = float(np.arccos(float(np.clip(momentum[2] / momentum_norm, -1.0, 1.0))))

    equatorial = node_norm <= _EQUATORIAL_TOLERANCE * momentum_norm
    circular = eccentricity <= _CIRCULAR_TOLERANCE

    if equatorial:
        raan = 0.0
        if circular:
            arg_perigee = 0.0
            true_anomaly = float(np.arctan2(r_vec[1], r_vec[0]))
        else:
            arg_perigee = float(np.arctan2(ecc_vec[1], ecc_vec[0]))
            true_anomaly = _signed_angle(ecc_vec, r_vec, momentum)
    else:
        raan = float(np.arctan2(node[1], node[0]))
        if circular:
            arg_perigee = 0.0
            true_anomaly = _signed_angle(node, r_vec, momentum)
        else:
            arg_perigee = _signed_angle(node, ecc_vec, momentum)
            true_anomaly = _signed_angle(ecc_vec, r_vec, momentum)

    return KeplerianElements(
        semi_major_axis_m=semi_major_axis,
        eccentricity=eccentricity,
        inclination_rad=inclination,
        raan_rad=raan % _TWO_PI,
        arg_perigee_rad=arg_perigee % _TWO_PI,
        true_anomaly_rad=true_anomaly % _TWO_PI,
        gravitational_parameter=mu,
    )


def _signed_angle(first: Vector, second: Vector, normal: Vector) -> float:
    """Return the angle from ``first`` to ``second`` measured about ``normal``."""
    cross = np.cross(first, second)
    sine = float(np.dot(cross, normal)) / float(np.linalg.norm(normal))
    cosine = float(np.dot(first, second))
    return float(np.arctan2(sine, cosine))
