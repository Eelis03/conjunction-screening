"""Array construction helpers shared by the model layer.

Model objects are frozen dataclasses that hold numpy arrays. These helpers copy
incoming data, check its shape, and mark the result read-only so that a frozen
object cannot be mutated through a reference to one of its arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["Matrix", "Vector", "as_matrix", "as_vector", "symmetrised", "unit_vector"]

Vector = npt.NDArray[np.float64]
"""A one-dimensional array of float64."""

Matrix = npt.NDArray[np.float64]
"""A two-dimensional array of float64."""


def as_vector(values: npt.ArrayLike, size: int, name: str) -> Vector:
    """Return ``values`` as a finite read-only float64 vector of length ``size``."""
    array = np.array(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def as_matrix(values: npt.ArrayLike, rows: int, columns: int, name: str) -> Matrix:
    """Return ``values`` as a finite read-only float64 array of shape ``(rows, columns)``."""
    array = np.array(values, dtype=np.float64)
    if array.shape != (rows, columns):
        raise ValueError(f"{name} must have shape ({rows}, {columns}), got {array.shape}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def unit_vector(vector: Vector, name: str) -> Vector:
    """Return ``vector`` scaled to unit length.

    Raises ``ValueError`` when the vector has no direction, which is always a
    modelling error rather than a numerical accident.
    """
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} has zero magnitude and therefore no direction")
    return np.asarray(vector, dtype=np.float64) / norm


def symmetrised(matrix: Matrix) -> Matrix:
    """Return ``(A + A.T) / 2``.

    Congruence transforms of a symmetric matrix are symmetric in exact
    arithmetic but pick up asymmetry of order machine epsilon in floating point.
    Applying this after every transform keeps eigenvalue routines on their
    symmetric code path.
    """
    array = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (array + array.T)
