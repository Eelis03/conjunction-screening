"""Covariance values and the checks that keep them physically meaningful.

A covariance matrix must stay symmetric and positive semi-definite through every
transform applied to it. The type in this module enforces symmetry on
construction and offers an explicit definiteness check that the test suite
applies after each stage of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from conjunction_screening.model.arrays import Matrix, Vector, as_matrix, symmetrised
from conjunction_screening.model.state import KeplerianElements, element_state_jacobian

__all__ = [
    "Covariance",
    "ElementSigmas",
    "combine_covariances",
    "covariance_from_element_sigmas",
    "is_symmetric_positive_semidefinite",
    "smallest_eigenvalue",
]

_SYMMETRY_TOLERANCE: Final[float] = 1e-9
"""Relative asymmetry accepted on construction.

Congruence transforms introduce asymmetry of order machine epsilon times the
matrix norm. The tolerance is relative to the largest absolute entry so that it
scales with the units the caller chose, and it is set well above epsilon so that
a chain of transforms does not trip it while a genuinely non-symmetric input does.
"""


@dataclass(frozen=True, slots=True)
class Covariance:
    """A symmetric covariance matrix with a named frame.

    Attributes:
        matrix: Square symmetric array. Units are the square of the units of the
            quantity described, so m^2 for a position covariance.
        frame: Free-form label recording the frame the matrix is expressed in,
            for example ``"ECI"``, ``"RIC"``, or ``"encounter"``. The label is
            not interpreted; it exists so that a mis-ordered pipeline is visible
            in a trace rather than silently wrong.
    """

    matrix: Matrix
    frame: str = "unspecified"

    def __post_init__(self) -> None:
        array = np.asarray(self.matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"covariance must be square, got shape {array.shape}")
        size = array.shape[0]
        checked = as_matrix(array, size, size, "matrix")
        scale = float(np.max(np.abs(checked))) if size else 0.0
        asymmetry = float(np.max(np.abs(checked - checked.T))) if size else 0.0
        if scale > 0.0 and asymmetry > _SYMMETRY_TOLERANCE * scale:
            raise ValueError(
                f"covariance is not symmetric: worst asymmetry {asymmetry:.3e} "
                f"against scale {scale:.3e}"
            )
        object.__setattr__(self, "matrix", as_matrix(symmetrised(checked), size, size, "matrix"))
        object.__setattr__(self, "frame", str(self.frame))

    @property
    def dimension(self) -> int:
        """Side length of the covariance matrix."""
        return int(self.matrix.shape[0])

    @property
    def standard_deviations(self) -> Matrix:
        """Square roots of the diagonal entries."""
        return np.sqrt(np.diag(self.matrix))

    def position_block(self) -> Covariance:
        """Return the leading 3 by 3 position block of a 6 by 6 state covariance."""
        if self.dimension != 6:
            raise ValueError(f"position_block requires a 6 by 6 covariance, got {self.dimension}")
        return Covariance(matrix=self.matrix[:3, :3], frame=self.frame)

    def scaled(self, factor: float) -> Covariance:
        """Return this covariance with every standard deviation multiplied by ``factor``.

        The matrix itself is multiplied by ``factor ** 2``. Scaling the standard
        deviation rather than the variance is the convention used in the
        dilution study, where the horizontal axis is a multiple of the nominal
        one sigma uncertainty.
        """
        if not factor > 0.0:
            raise ValueError("covariance scale factor must be positive")
        return Covariance(matrix=self.matrix * factor**2, frame=self.frame)

    def transformed(self, rotation: Matrix, frame: str) -> Covariance:
        """Return ``R C R^T`` for a rotation or projection ``R``."""
        operator = np.asarray(rotation, dtype=np.float64)
        if operator.ndim != 2 or operator.shape[1] != self.dimension:
            raise ValueError(
                f"operator with shape {operator.shape} cannot act on a "
                f"{self.dimension} by {self.dimension} covariance"
            )
        return Covariance(matrix=symmetrised(operator @ self.matrix @ operator.T), frame=frame)


def combine_covariances(first: Covariance, second: Covariance) -> Covariance:
    """Return the covariance of the difference of two independent states.

    The relative position of two objects whose errors are uncorrelated has the
    sum of their covariances. Correlated errors, which arise when both objects
    are tracked by the same sensor network and share atmospheric density model
    error, would require a joint covariance that this library does not model.
    """
    if first.dimension != second.dimension:
        raise ValueError("covariances must have equal dimension to combine")
    if first.frame != second.frame:
        raise ValueError(
            f"covariances must share a frame to combine, got {first.frame!r} and {second.frame!r}"
        )
    return Covariance(matrix=first.matrix + second.matrix, frame=first.frame)


def smallest_eigenvalue(covariance: Covariance) -> float:
    """Return the smallest eigenvalue of a covariance matrix."""
    return float(np.min(np.linalg.eigvalsh(covariance.matrix)))


def is_symmetric_positive_semidefinite(
    covariance: Covariance, relative_tolerance: float = 1e-12
) -> bool:
    """Report whether ``covariance`` is symmetric and positive semi-definite.

    Args:
        covariance: Matrix under test.
        relative_tolerance: Allowance on the smallest eigenvalue, expressed as a
            fraction of the largest eigenvalue. A congruence transform of a
            semi-definite matrix can produce a smallest eigenvalue of order
            machine epsilon times the largest one with either sign, so a check
            against exactly zero would be a check on rounding rather than on the
            mathematics.

    Returns:
        True when the matrix is symmetric to the construction tolerance and its
        smallest eigenvalue exceeds ``-relative_tolerance`` times its largest.
    """
    matrix = covariance.matrix
    if not bool(np.allclose(matrix, matrix.T, rtol=0.0, atol=_SYMMETRY_TOLERANCE * _scale(matrix))):
        return False
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    return bool(float(np.min(eigenvalues)) >= -relative_tolerance * max(largest, 1.0))


def _scale(matrix: Matrix) -> float:
    value = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    return value if value > 0.0 else 1.0


@dataclass(frozen=True, slots=True)
class ElementSigmas:
    """Uncorrelated one sigma uncertainties on the classical orbital elements.

    Attributes:
        semi_major_axis_m: Uncertainty on the semi-major axis, in m.
        eccentricity: Uncertainty on the eccentricity, dimensionless.
        inclination_rad: Uncertainty on the inclination, in rad.
        raan_rad: Uncertainty on the right ascension of the ascending node.
        arg_perigee_rad: Uncertainty on the argument of perigee.
        mean_anomaly_rad: Uncertainty on the mean anomaly at epoch.
    """

    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    mean_anomaly_rad: float

    def as_vector(self) -> Vector:
        """Return the six sigmas in the order used by the element Jacobian."""
        return np.array(
            [
                self.semi_major_axis_m,
                self.eccentricity,
                self.inclination_rad,
                self.raan_rad,
                self.arg_perigee_rad,
                self.mean_anomaly_rad,
            ],
            dtype=np.float64,
        )


def covariance_from_element_sigmas(
    elements: KeplerianElements, sigmas: ElementSigmas
) -> Covariance:
    """Map uncorrelated element uncertainties into an inertial state covariance.

    The result is ``J diag(sigma^2) J^T`` with ``J`` the Jacobian of the state
    with respect to the elements. Written that way it is a sum of positive
    multiples of rank one outer products, so it is positive semi-definite by
    construction rather than by luck.

    Building a synthetic covariance in element space rather than directly in the
    RIC frame matters for a screening study. A diagonal RIC covariance implies an
    uncorrelated semi-major axis error of the same order as the radial position
    error, and a semi-major axis error drives in-track position error linearly in
    time at a rate of three halves of the mean motion. Over a one day screening
    window that produces an in-track uncertainty an order of magnitude larger than
    an operational covariance shows. Specifying the semi-major axis uncertainty
    directly puts that growth under control and makes it visible.
    """
    jacobian = element_state_jacobian(elements)
    variances = sigmas.as_vector() ** 2
    matrix = jacobian @ np.diag(variances) @ jacobian.T
    return Covariance(matrix=symmetrised(matrix), frame="ECI")
