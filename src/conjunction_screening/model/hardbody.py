"""Hard body geometry beyond the sphere.

The collision condition under linear relative motion is that the straight line
the secondary travels along passes through the combined body. Projecting that
line onto the encounter plane, which is normal to it, turns the condition into
"the piercing point falls inside the shadow the combined body casts along the
relative velocity". For two spheres the shadow is a disc of the combined radius
and the direction does not matter, which is why the sphere model needs no
geometry beyond one number.

For anything else the direction matters, and this module supplies the three
pieces that makes it computable:

* ``HardBody`` holds a convex body as an ellipsoid ``{x : x^T S^-1 x <= 1}``.
  A sphere of radius ``R`` is ``S = R^2 I``, so the sphere is a special case
  rather than a separate path.
* ``combine_hard_bodies`` replaces the pair by one body that contains their
  Minkowski sum, which is the body the relative position vector must miss. The
  sum of two ellipsoids is not an ellipsoid, so an outer approximation is used;
  for two spheres it is exact and returns the combined radius.
* ``projected_cross_section`` casts the shadow. The shadow of an ellipsoid is an
  ellipse, and its shape matrix is ``B S B^T`` for the same 2 by 3 projection
  ``B`` that carries the covariance into the plane, because both are images of a
  quadratic form under a linear map.

The whole construction reduces to the disc when both bodies are spheres, so
nothing an existing caller does changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from conjunction_screening.model.arrays import Matrix, Vector, as_matrix, symmetrised

__all__ = [
    "CrossSection",
    "HardBody",
    "combine_hard_bodies",
    "projected_cross_section",
]

_ISOTROPY_TOLERANCE: Final[float] = 1e-12
"""Relative spread of the semi-axes below which a body counts as a sphere.

A combined body assembled from two spheres reaches ``(R1 + R2)^2 I`` up to
rounding rather than exactly, so a test against exact equality would answer a
question about floating point rather than about geometry.
"""


@dataclass(frozen=True, slots=True)
class HardBody:
    """A convex hard body, modelled as an ellipsoid in the inertial frame.

    Attributes:
        shape_matrix: Symmetric positive definite 3 by 3 array ``S``, with the
            body being ``{x : x^T S^-1 x <= 1}``. Its eigenvalues are the squares
            of the semi-axes and its eigenvectors are the body axes. Units are
            m^2, so that ``S`` transforms exactly as a covariance does.
    """

    shape_matrix: Matrix

    def __post_init__(self) -> None:
        checked = as_matrix(symmetrised(self.shape_matrix), 3, 3, "shape_matrix")
        if float(np.min(np.linalg.eigvalsh(checked))) <= 0.0:
            raise ValueError("shape_matrix must be positive definite; a flat body has no shadow")
        object.__setattr__(self, "shape_matrix", checked)

    @classmethod
    def sphere(cls, radius_m: float) -> HardBody:
        """Return the sphere of radius ``radius_m``."""
        if not radius_m > 0.0:
            raise ValueError("radius_m must be positive")
        return cls(shape_matrix=np.eye(3, dtype=np.float64) * radius_m**2)

    @classmethod
    def ellipsoid(
        cls, semi_axes_m: tuple[float, float, float], orientation: Matrix | None = None
    ) -> HardBody:
        """Return an ellipsoid with the given semi-axes.

        Args:
            semi_axes_m: The three semi-axis lengths, in m.
            orientation: Rotation whose columns are the body axes expressed in the
                inertial frame. The identity is used when it is omitted, which
                puts the semi-axes on the inertial axes.

        Raises:
            ValueError: If a semi-axis is not positive, or if ``orientation`` is
                not orthogonal. A non-orthogonal matrix would shear the body and
                silently change its size.
        """
        axes = np.asarray(semi_axes_m, dtype=np.float64)
        if axes.shape != (3,) or not bool(np.all(axes > 0.0)):
            raise ValueError("semi_axes_m must be three positive lengths")
        if orientation is None:
            rotation = np.eye(3, dtype=np.float64)
        else:
            rotation = as_matrix(orientation, 3, 3, "orientation")
        residual = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
        if residual > 1e-10:
            raise ValueError(f"orientation must be orthogonal, worst residual {residual:.3e}")
        return cls(shape_matrix=symmetrised(rotation @ np.diag(axes**2) @ rotation.T))

    @property
    def semi_axes_m(self) -> Vector:
        """The three semi-axis lengths in decreasing order, in m."""
        eigenvalues = np.linalg.eigvalsh(self.shape_matrix)
        return np.asarray(np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1], dtype=np.float64)

    @property
    def bounding_radius_m(self) -> float:
        """Radius of the smallest sphere containing this body, in m."""
        return float(self.semi_axes_m[0])

    @property
    def is_sphere(self) -> bool:
        """True when every semi-axis is the same to within rounding."""
        axes = self.semi_axes_m
        return bool(axes[0] - axes[2] <= _ISOTROPY_TOLERANCE * axes[0])

    def support_m(self, direction: Vector) -> float:
        """Return how far the body reaches in ``direction``, in m.

        The support function of the ellipsoid is ``sqrt(l^T S l)`` for a unit
        ``l``. It is the quantity that makes containment checkable: one convex
        body contains another exactly when its support is at least as large in
        every direction.
        """
        unit = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(unit))
        if norm == 0.0:
            raise ValueError("direction has zero magnitude")
        unit = unit / norm
        return float(np.sqrt(unit @ self.shape_matrix @ unit))


def combine_hard_bodies(first: HardBody, second: HardBody) -> HardBody:
    """Return one body containing the Minkowski sum of ``first`` and ``second``.

    The collision condition involves the sum of the two bodies, because the
    relative position vector must clear both. The Minkowski sum of two ellipsoids
    is not an ellipsoid, but the family

        S(p) = (1 + 1/p) S1 + (1 + p) S2,   p > 0

    contains it for every ``p``. That follows from the support functions: with
    ``a = sqrt(l^T S1 l)`` and ``b = sqrt(l^T S2 l)`` the difference between the
    squared supports is

        (1 + 1/p) a^2 + (1 + p) b^2 - (a + b)^2 = (a / sqrt(p) - b sqrt(p))^2

    which is non-negative, and zero when ``p = a / b``. No single ``p`` is tight
    in every direction unless the two bodies are similar, so ``p`` is chosen to
    minimise the trace, which is the sum of the squared semi-axes. For two
    spheres that choice is ``R1 / R2``, at which the result is exactly the sphere
    of radius ``R1 + R2``, so the combined radius convention is recovered rather
    than approximated.
    """
    first_trace = float(np.trace(first.shape_matrix))
    second_trace = float(np.trace(second.shape_matrix))
    ratio = float(np.sqrt(first_trace / second_trace))
    combined = (1.0 + 1.0 / ratio) * first.shape_matrix + (1.0 + ratio) * second.shape_matrix
    return HardBody(shape_matrix=symmetrised(combined))


@dataclass(frozen=True, slots=True)
class CrossSection:
    """The hard body as it is seen in the encounter plane.

    Attributes:
        matrix: Symmetric positive definite 2 by 2 shape matrix ``M`` of the
            ellipse ``{p : p^T M^-1 p <= 1}``, in m^2, expressed in the same
            in-plane coordinates as the miss vector.
    """

    matrix: Matrix

    def __post_init__(self) -> None:
        checked = as_matrix(symmetrised(self.matrix), 2, 2, "matrix")
        if float(np.min(np.linalg.eigvalsh(checked))) <= 0.0:
            raise ValueError("cross section matrix must be positive definite")
        object.__setattr__(self, "matrix", checked)

    @classmethod
    def disc(cls, radius_m: float) -> CrossSection:
        """Return the disc of radius ``radius_m``, which is what a sphere casts."""
        if not radius_m > 0.0:
            raise ValueError("radius_m must be positive")
        return cls(matrix=np.eye(2, dtype=np.float64) * radius_m**2)

    @classmethod
    def ellipse(cls, major_m: float, minor_m: float, orientation_rad: float = 0.0) -> CrossSection:
        """Return an ellipse with the given semi-axes.

        Args:
            major_m: Semi-axis along the direction ``orientation_rad``, in m.
            minor_m: Semi-axis perpendicular to it, in m.
            orientation_rad: Angle from the first in-plane axis, which is the
                miss direction, to the major semi-axis.
        """
        if not major_m > 0.0 or not minor_m > 0.0:
            raise ValueError("both semi-axes must be positive")
        cosine, sine = float(np.cos(orientation_rad)), float(np.sin(orientation_rad))
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        return cls(matrix=symmetrised(rotation @ np.diag([major_m**2, minor_m**2]) @ rotation.T))

    @property
    def semi_axes_m(self) -> tuple[float, float]:
        """Major and minor semi-axes of the ellipse, in m."""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return float(np.sqrt(eigenvalues[1])), float(np.sqrt(eigenvalues[0]))

    @property
    def is_circular(self) -> bool:
        """True when the two semi-axes agree to within rounding."""
        major, minor = self.semi_axes_m
        return bool(major - minor <= _ISOTROPY_TOLERANCE * major)

    @property
    def equivalent_radius_m(self) -> float:
        """Radius of the disc of the same area, ``sqrt(a b)``, in m."""
        major, minor = self.semi_axes_m
        return float(np.sqrt(major * minor))

    @property
    def area_m2(self) -> float:
        """Area of the cross section, in m^2."""
        major, minor = self.semi_axes_m
        return float(np.pi * major * minor)

    def radius_at(self, angle_rad: float) -> float:
        """Return the distance from the centre to the boundary along ``angle_rad``.

        For the direction ``u`` at that angle the boundary point is ``r u`` with
        ``r^2 u^T M^-1 u = 1``. The 2 by 2 inverse is written out rather than
        formed, because this is the inner limit of a quadrature and is evaluated
        many thousands of times.
        """
        (m11, m12), (_, m22) = self.matrix
        determinant = float(m11 * m22 - m12 * m12)
        cosine, sine = float(np.cos(angle_rad)), float(np.sin(angle_rad))
        quadratic = float(m22 * cosine**2 - 2.0 * m12 * cosine * sine + m11 * sine**2)
        return float(np.sqrt(determinant / quadratic))

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Return which of the given in-plane points fall inside the cross section.

        Args:
            points: Array of shape ``(n, 2)`` of in-plane coordinates, in m.
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"points must have shape (n, 2), got {array.shape}")
        (m11, m12), (_, m22) = self.matrix
        determinant = float(m11 * m22 - m12 * m12)
        x, y = array[:, 0], array[:, 1]
        quadratic = m22 * x**2 - 2.0 * m12 * x * y + m11 * y**2
        return np.asarray(quadratic <= determinant, dtype=np.bool_)

    def scaled(self, factor: float) -> CrossSection:
        """Return this cross section with both semi-axes multiplied by ``factor``."""
        if not factor > 0.0:
            raise ValueError("cross section scale factor must be positive")
        return CrossSection(matrix=self.matrix * factor**2)

    def rotated(self, rotation: Matrix) -> CrossSection:
        """Return this cross section expressed in the frame whose axes are ``rotation``.

        ``rotation`` has the new basis vectors as its columns, which is the
        convention a symmetric eigendecomposition returns, so the shape matrix
        transforms as ``R^T M R``.
        """
        operator = as_matrix(rotation, 2, 2, "rotation")
        return CrossSection(matrix=symmetrised(operator.T @ self.matrix @ operator))


def projected_cross_section(body: HardBody, basis: Matrix) -> CrossSection:
    """Return the shadow ``body`` casts on the plane spanned by ``basis``.

    Writing the ellipsoid as ``{L u : |u| <= 1}`` with ``S = L L^T``, its image
    under the projection ``B`` is ``{B L u : |u| <= 1}``, an ellipse with shape
    matrix ``(B L)(B L)^T = B S B^T``. The projection is along the plane normal,
    which for the encounter plane is the relative velocity, so this is exactly
    the outline the secondary sees as it approaches.

    Args:
        body: The body casting the shadow.
        basis: The 2 by 3 encounter plane projection, with orthonormal rows.
    """
    operator = as_matrix(basis, 2, 3, "basis")
    return CrossSection(matrix=symmetrised(operator @ body.shape_matrix @ operator.T))
