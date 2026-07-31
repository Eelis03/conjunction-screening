"""Frame rotations between the inertial frame and the RIC frame.

RIC, also written RTN, is the local orbital frame of a single object:

* radial, along the position vector;
* in-track, completing the right-handed set and lying along the velocity for a
  circular orbit;
* cross-track, along the orbital angular momentum.

Covariances supplied in conjunction data messages are quoted in this frame, and
the encounter plane construction reads more naturally in it, so the pipeline
carries both representations.
"""

from __future__ import annotations

import numpy as np

from conjunction_screening.model.arrays import Matrix, unit_vector
from conjunction_screening.model.covariance import Covariance
from conjunction_screening.model.state import OrbitState

__all__ = [
    "block_diagonal_rotation",
    "inertial_to_ric_rotation",
    "rotate_covariance_to_inertial",
    "rotate_covariance_to_ric",
]


def inertial_to_ric_rotation(state: OrbitState) -> Matrix:
    """Return the rotation whose rows are the RIC axes expressed in the inertial frame.

    Applying the result to an inertial vector gives its RIC components.
    """
    radial = unit_vector(np.asarray(state.position_m, dtype=np.float64), "position_m")
    momentum = np.cross(state.position_m, state.velocity_m_s)
    cross_track = unit_vector(momentum, "angular momentum")
    in_track = np.cross(cross_track, radial)
    return np.stack((radial, in_track, cross_track), axis=0)


def block_diagonal_rotation(rotation: Matrix) -> Matrix:
    """Lift a 3 by 3 rotation to the 6 by 6 rotation acting on position and velocity.

    The lift is valid for a rotation held fixed at one epoch, which is how a
    conjunction data message covariance is defined: the RIC axes are frozen at
    the epoch of the message rather than rotating with the object.
    """
    lifted = np.zeros((6, 6), dtype=np.float64)
    lifted[:3, :3] = rotation
    lifted[3:, 3:] = rotation
    return lifted


def rotate_covariance_to_ric(covariance: Covariance, state: OrbitState) -> Covariance:
    """Rotate an inertial covariance into the RIC frame of ``state``."""
    rotation = inertial_to_ric_rotation(state)
    operator = rotation if covariance.dimension == 3 else block_diagonal_rotation(rotation)
    return covariance.transformed(operator, frame="RIC")


def rotate_covariance_to_inertial(covariance: Covariance, state: OrbitState) -> Covariance:
    """Rotate a RIC covariance into the inertial frame of ``state``."""
    rotation = inertial_to_ric_rotation(state).T
    operator = rotation if covariance.dimension == 3 else block_diagonal_rotation(rotation)
    return covariance.transformed(operator, frame="ECI")
