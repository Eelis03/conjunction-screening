"""Encounter plane geometry.

The two-dimensional probability of collision is defined on the encounter plane,
the plane through the primary that is normal to the relative velocity at the time
of closest approach. Under the linear relative motion assumption the secondary
travels in a straight line through that plane, so the three-dimensional problem
of "does the straight line pass within the combined object radius" collapses to
the two-dimensional problem of "does the piercing point fall inside a disc of
that radius". This module builds the plane, projects the relative position and
its covariance into it, and records the quantities the probability methods need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from conjunction_screening.model.arrays import Matrix, Vector, as_matrix, as_vector, unit_vector
from conjunction_screening.model.covariance import Covariance
from conjunction_screening.model.hardbody import CrossSection

__all__ = [
    "EncounterGeometry",
    "PrincipalForm",
    "encounter_plane_basis",
    "planar_encounter",
    "principal_axis_form",
    "project_to_encounter_plane",
]

_PARALLEL_TOLERANCE: Final[float] = 1e-12
"""Relative magnitude below which a projected relative position has no direction."""

_RADIUS_CONSISTENCY_TOLERANCE: Final[float] = 1e-9
"""Relative agreement required between a cross section and the reported radius.

``hard_body_radius_m`` is the number every report prints and every closed form
uses. When a cross section is attached it must be the area-equivalent radius of
that cross section, or the two would describe different bodies and a reader
could not tell which one produced the probability.
"""


def encounter_plane_basis(relative_position_m: Vector, relative_velocity_m_s: Vector) -> Matrix:
    """Return the 2 by 3 projection onto the encounter plane.

    Row 0 is the in-plane axis along the projected relative position, so the
    projected miss vector is ``(d, 0)`` with ``d`` non-negative. Row 1 completes
    a right-handed set with the relative velocity direction. Choosing the first
    axis along the miss vector is the convention in Foster and Estes and removes
    one free parameter from the probability integral.

    Args:
        relative_position_m: Secondary position minus primary position, in m.
        relative_velocity_m_s: Secondary velocity minus primary velocity, in m/s.

    Returns:
        Array of shape ``(2, 3)`` whose rows are orthonormal and orthogonal to
        the relative velocity.
    """
    position = as_vector(relative_position_m, 3, "relative_position_m")
    velocity = as_vector(relative_velocity_m_s, 3, "relative_velocity_m_s")
    normal = unit_vector(velocity, "relative_velocity_m_s")

    in_plane = position - float(np.dot(position, normal)) * normal
    magnitude = float(np.linalg.norm(in_plane))
    reference = float(np.linalg.norm(position))
    if magnitude <= _PARALLEL_TOLERANCE * max(reference, 1.0):
        # A head-on geometry with no lateral offset. Any in-plane direction is a
        # valid first axis; take the one furthest from the normal for conditioning.
        fallback = np.zeros(3, dtype=np.float64)
        fallback[int(np.argmin(np.abs(normal)))] = 1.0
        in_plane = fallback - float(np.dot(fallback, normal)) * normal
    first = unit_vector(in_plane, "projected relative position")
    second = np.cross(normal, first)
    return np.stack((first, second), axis=0)


def project_to_encounter_plane(vector: Vector, basis: Matrix) -> Vector:
    """Project a three-dimensional vector onto the encounter plane."""
    operator = as_matrix(basis, 2, 3, "basis")
    return np.asarray(operator @ np.asarray(vector, dtype=np.float64), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class EncounterGeometry:
    """Everything a two-dimensional probability method needs about one conjunction.

    Attributes:
        tca_s: Time of closest approach, in s from the screening epoch.
        relative_position_m: Secondary minus primary position at the time of
            closest approach, in the inertial frame, in m.
        relative_velocity_m_s: Secondary minus primary velocity at the time of
            closest approach, in the inertial frame, in m/s.
        relative_covariance: Combined 3 by 3 relative position covariance in the
            inertial frame at the time of closest approach, in m^2.
        relative_covariance_ric: The same covariance in the RIC frame of the
            primary at the time of closest approach. Carried for reporting; the
            probability methods do not read it.
        basis: The 2 by 3 encounter plane projection.
        miss_vector_m: Relative position projected into the plane, in m. Its
            first component is the projected miss distance and its second is zero
            by construction of the basis.
        plane_covariance: Combined relative covariance projected into the plane,
            a 2 by 2 matrix in m^2.
        hard_body_radius_m: Radius of the disc the collision is judged against,
            in m. For two spheres it is the sum of the two radii. When
            ``cross_section`` is set it is that section's area-equivalent radius.
        cross_section: Shadow the combined hard body casts on the encounter
            plane, in the same in-plane coordinates as ``miss_vector_m``. None
            means the shadow is the disc of ``hard_body_radius_m``, which is what
            two spheres cast from any direction.
    """

    tca_s: float
    relative_position_m: Vector
    relative_velocity_m_s: Vector
    relative_covariance: Covariance
    relative_covariance_ric: Covariance
    basis: Matrix
    miss_vector_m: Vector
    plane_covariance: Covariance
    hard_body_radius_m: float
    cross_section: CrossSection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tca_s", float(self.tca_s))
        object.__setattr__(
            self, "relative_position_m", as_vector(self.relative_position_m, 3, "relative_position")
        )
        object.__setattr__(
            self,
            "relative_velocity_m_s",
            as_vector(self.relative_velocity_m_s, 3, "relative_velocity"),
        )
        object.__setattr__(self, "basis", as_matrix(self.basis, 2, 3, "basis"))
        object.__setattr__(self, "miss_vector_m", as_vector(self.miss_vector_m, 2, "miss_vector"))
        object.__setattr__(self, "hard_body_radius_m", float(self.hard_body_radius_m))
        if not self.hard_body_radius_m > 0.0:
            raise ValueError("hard_body_radius_m must be positive")
        if self.relative_covariance.dimension != 3:
            raise ValueError("relative_covariance must be 3 by 3")
        if self.plane_covariance.dimension != 2:
            raise ValueError("plane_covariance must be 2 by 2")
        section = self.cross_section
        if section is not None:
            equivalent = section.equivalent_radius_m
            if abs(equivalent - self.hard_body_radius_m) > (
                _RADIUS_CONSISTENCY_TOLERANCE * equivalent
            ):
                raise ValueError(
                    f"hard_body_radius_m {self.hard_body_radius_m:.6e} does not match the "
                    f"area-equivalent radius {equivalent:.6e} of the cross section"
                )

    @property
    def miss_distance_m(self) -> float:
        """Three-dimensional miss distance at the time of closest approach, in m."""
        return float(np.linalg.norm(self.relative_position_m))

    @property
    def projected_miss_distance_m(self) -> float:
        """Miss distance measured in the encounter plane, in m."""
        return float(np.linalg.norm(self.miss_vector_m))

    @property
    def relative_speed_m_s(self) -> float:
        """Relative speed at the time of closest approach, in m/s."""
        return float(np.linalg.norm(self.relative_velocity_m_s))

    def with_scaled_covariance(self, factor: float) -> EncounterGeometry:
        """Return a copy with every covariance standard deviation multiplied by ``factor``."""
        return EncounterGeometry(
            tca_s=self.tca_s,
            relative_position_m=self.relative_position_m,
            relative_velocity_m_s=self.relative_velocity_m_s,
            relative_covariance=self.relative_covariance.scaled(factor),
            relative_covariance_ric=self.relative_covariance_ric.scaled(factor),
            basis=self.basis,
            miss_vector_m=self.miss_vector_m,
            plane_covariance=self.plane_covariance.scaled(factor),
            hard_body_radius_m=self.hard_body_radius_m,
            cross_section=self.cross_section,
        )

    def with_hard_body_radius(self, radius_m: float) -> EncounterGeometry:
        """Return a copy with a different combined hard body radius.

        A cross section is carried over with its shape held and its size changed,
        so that this stays a statement about how big the body is and not about
        what shape it is.
        """
        section = self.cross_section
        if section is not None:
            section = section.scaled(radius_m / section.equivalent_radius_m)
        return EncounterGeometry(
            tca_s=self.tca_s,
            relative_position_m=self.relative_position_m,
            relative_velocity_m_s=self.relative_velocity_m_s,
            relative_covariance=self.relative_covariance,
            relative_covariance_ric=self.relative_covariance_ric,
            basis=self.basis,
            miss_vector_m=self.miss_vector_m,
            plane_covariance=self.plane_covariance,
            hard_body_radius_m=radius_m,
            cross_section=section,
        )

    def with_cross_section(self, section: CrossSection) -> EncounterGeometry:
        """Return a copy whose hard body casts ``section`` on the encounter plane.

        The reported hard body radius follows the section, so the two can never
        disagree.
        """
        return EncounterGeometry(
            tca_s=self.tca_s,
            relative_position_m=self.relative_position_m,
            relative_velocity_m_s=self.relative_velocity_m_s,
            relative_covariance=self.relative_covariance,
            relative_covariance_ric=self.relative_covariance_ric,
            basis=self.basis,
            miss_vector_m=self.miss_vector_m,
            plane_covariance=self.plane_covariance,
            hard_body_radius_m=section.equivalent_radius_m,
            cross_section=section,
        )


@dataclass(frozen=True, slots=True)
class PrincipalForm:
    """The encounter reduced to the principal axes of its in-plane covariance.

    All three analytic probability methods start from this form, in which the
    covariance is diagonal and the integral separates.

    Attributes:
        sigma_x_m: Larger principal standard deviation, in m.
        sigma_y_m: Smaller principal standard deviation, in m.
        mean_x_m: Miss vector component along the first principal axis, in m.
        mean_y_m: Miss vector component along the second principal axis, in m.
        radius_m: Hard body radius, in m.
        cross_section: Hard body shadow rotated into the same principal axes, or
            None when it is the disc of ``radius_m``.
    """

    sigma_x_m: float
    sigma_y_m: float
    mean_x_m: float
    mean_y_m: float
    radius_m: float
    cross_section: CrossSection | None = None

    @property
    def normalised_miss_distance(self) -> float:
        """Miss distance measured in standard deviations of the combined covariance.

        The probability depends on the miss vector only through this quantity and
        the ratio of the hard body area to the covariance area, so two events
        that differ in metres can be ordered the other way once their
        uncertainties are taken into account. It is the Mahalanobis distance of
        the miss vector, dimensionless, and unlike the two components it does not
        depend on which principal axis was called first.
        """
        return float(
            np.hypot(self.mean_x_m / self.sigma_x_m, self.mean_y_m / self.sigma_y_m)
        )


def planar_encounter(
    miss_distance_m: float,
    sigma_x_m: float,
    sigma_y_m: float,
    hard_body_radius_m: float,
    orientation_rad: float = 0.0,
    relative_speed_m_s: float = 1.0e4,
    along_velocity_sigma_m: float | None = None,
    tca_s: float = 0.0,
) -> EncounterGeometry:
    """Build an encounter directly from its encounter plane parameters.

    Useful when the quantity of interest is the probability integral rather than
    the orbit that produced it: tests and method comparisons need encounters with
    exactly known miss distance, principal standard deviations, and orientation,
    which is awkward to arrange by choosing two orbits.

    The relative velocity is placed along the third inertial axis, so the
    encounter plane is the first two axes and the covariance rotation is applied
    inside that plane. The variance along the relative velocity does not enter the
    two-dimensional formulation, and is set to the geometric mean of the two
    in-plane variances unless given.

    Args:
        miss_distance_m: Projected miss distance, in m.
        sigma_x_m: First principal standard deviation of the in-plane covariance.
        sigma_y_m: Second principal standard deviation.
        hard_body_radius_m: Combined hard body radius, in m.
        orientation_rad: Angle from the miss direction to the first principal axis.
        relative_speed_m_s: Relative speed at closest approach, in m/s.
        along_velocity_sigma_m: Standard deviation along the relative velocity.
        tca_s: Time of closest approach recorded on the result, in s.

    Returns:
        An encounter geometry whose principal axis form reproduces the arguments.
    """
    if not sigma_x_m > 0.0 or not sigma_y_m > 0.0:
        raise ValueError("both principal standard deviations must be positive")
    cosine, sine = float(np.cos(orientation_rad)), float(np.sin(orientation_rad))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    plane_matrix = rotation @ np.diag([sigma_x_m**2, sigma_y_m**2]) @ rotation.T

    along = along_velocity_sigma_m
    if along is None:
        along = float(np.sqrt(sigma_x_m * sigma_y_m))
    spatial = np.zeros((3, 3), dtype=np.float64)
    spatial[:2, :2] = plane_matrix
    spatial[2, 2] = along**2

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    relative_covariance = Covariance(matrix=spatial, frame="ECI")
    return EncounterGeometry(
        tca_s=tca_s,
        relative_position_m=np.array([miss_distance_m, 0.0, 0.0], dtype=np.float64),
        relative_velocity_m_s=np.array([0.0, 0.0, relative_speed_m_s], dtype=np.float64),
        relative_covariance=relative_covariance,
        relative_covariance_ric=Covariance(matrix=spatial, frame="RIC"),
        basis=basis,
        miss_vector_m=np.array([miss_distance_m, 0.0], dtype=np.float64),
        plane_covariance=Covariance(matrix=plane_matrix, frame="encounter"),
        hard_body_radius_m=hard_body_radius_m,
    )


def principal_axis_form(encounter: EncounterGeometry) -> PrincipalForm:
    """Rotate the in-plane covariance to its principal axes.

    The rotation is a symmetric eigendecomposition of a 2 by 2 matrix, so it is
    exact up to rounding and preserves both the miss distance and the covariance
    determinant.

    Raises:
        ValueError: If either principal variance is not strictly positive. A
            singular in-plane covariance gives an improper density and the
            two-dimensional probability is undefined.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(encounter.plane_covariance.matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if float(eigenvalues[1]) <= 0.0:
        raise ValueError(
            "in-plane covariance is singular; the two-dimensional probability is undefined"
        )
    mean = eigenvectors.T @ np.asarray(encounter.miss_vector_m, dtype=np.float64)
    section = encounter.cross_section
    return PrincipalForm(
        sigma_x_m=float(np.sqrt(eigenvalues[0])),
        sigma_y_m=float(np.sqrt(eigenvalues[1])),
        mean_x_m=float(mean[0]),
        mean_y_m=float(mean[1]),
        radius_m=encounter.hard_body_radius_m,
        cross_section=None if section is None else section.rotated(eigenvectors),
    )
