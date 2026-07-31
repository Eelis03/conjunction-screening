"""Two-body Keplerian propagation and linear covariance propagation.

Propagation is analytic: the state is converted to elements once, the mean
anomaly is advanced, and Kepler's equation is inverted. There is no numerical
integration, so the propagated state carries no step-size error and repeated
propagation over a long window accumulates no drift.

The state transition matrix is obtained by central differences of that analytic
propagator. Because the underlying flow is Hamiltonian the resulting matrix must
be symplectic, which gives the test suite an invariant that checks the
differencing step size, the propagator, and the element conversions at once.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import numpy as np

from conjunction_screening.model.arrays import Matrix, symmetrised
from conjunction_screening.model.constants import MU_EARTH
from conjunction_screening.model.covariance import Covariance
from conjunction_screening.model.state import (
    KeplerianElements,
    OrbitState,
    eccentric_to_mean_anomaly,
    eccentric_to_true_anomaly,
    elements_from_state,
    perifocal_to_inertial,
    solve_kepler_equation,
    state_from_elements,
    true_to_eccentric_anomaly,
)

__all__ = [
    "advance_elements",
    "propagate",
    "propagate_covariance",
    "propagate_many",
    "propagate_to",
    "state_transition_matrix",
    "symplectic_form",
    "symplectic_residual",
]

_STM_RELATIVE_STEP: Final[float] = 1e-6
"""Relative perturbation used for the central-difference state transition matrix.

Central differencing has truncation error of order ``h^2`` and round-off error of
order ``eps / h``, both relative. Balancing them puts the optimum near
``eps ** (1/3)`` which is 6e-6; 1e-6 sits close to that optimum and gives a
matrix accurate to roughly 1e-10 relative.
"""


def propagate(
    state: OrbitState, delta_time_s: float, gravitational_parameter: float = MU_EARTH
) -> OrbitState:
    """Advance a state by ``delta_time_s`` under two-body motion."""
    elements = elements_from_state(state, gravitational_parameter)
    advanced = advance_elements(elements, delta_time_s)
    return state_from_elements(advanced, epoch_s=state.epoch_s + float(delta_time_s))


def propagate_to(
    state: OrbitState, epoch_s: float, gravitational_parameter: float = MU_EARTH
) -> OrbitState:
    """Advance a state to an absolute epoch under two-body motion."""
    return propagate(state, float(epoch_s) - state.epoch_s, gravitational_parameter)


def advance_elements(elements: KeplerianElements, delta_time_s: float) -> KeplerianElements:
    """Advance the true anomaly of ``elements`` by a time interval."""
    ecc = elements.eccentricity
    mean_start = float(
        eccentric_to_mean_anomaly(true_to_eccentric_anomaly(elements.true_anomaly_rad, ecc), ecc)
    )
    mean_end = mean_start + elements.mean_motion_rad_s * float(delta_time_s)
    ecc_anomaly = solve_kepler_equation(np.asarray(mean_end, dtype=np.float64), ecc)
    true_anomaly = float(eccentric_to_true_anomaly(ecc_anomaly, ecc))
    return replace(elements, true_anomaly_rad=true_anomaly % (2.0 * np.pi))


def propagate_many(
    state: OrbitState,
    delta_times_s: np.ndarray,
    gravitational_parameter: float = MU_EARTH,
) -> tuple[Matrix, Matrix]:
    """Propagate one state to many offsets from its epoch.

    Args:
        state: State to propagate.
        delta_times_s: Offsets from ``state.epoch_s``, shape ``(n,)``.
        gravitational_parameter: Central body mu in m^3 / s^2.

    Returns:
        Positions of shape ``(n, 3)`` in m and velocities of shape ``(n, 3)`` in m/s.
    """
    elements = elements_from_state(state, gravitational_parameter)
    ecc = elements.eccentricity
    offsets = np.asarray(delta_times_s, dtype=np.float64)
    mean_start = float(
        eccentric_to_mean_anomaly(true_to_eccentric_anomaly(elements.true_anomaly_rad, ecc), ecc)
    )
    mean = mean_start + elements.mean_motion_rad_s * offsets
    ecc_anomaly = solve_kepler_equation(mean, ecc)
    true_anomaly = eccentric_to_true_anomaly(ecc_anomaly, ecc)

    semi_latus = elements.semi_latus_rectum_m
    radius = semi_latus / (1.0 + ecc * np.cos(true_anomaly))
    factor = float(np.sqrt(gravitational_parameter / semi_latus))
    position_pf = np.stack(
        (radius * np.cos(true_anomaly), radius * np.sin(true_anomaly), np.zeros_like(radius)),
        axis=-1,
    )
    velocity_pf = np.stack(
        (
            -factor * np.sin(true_anomaly),
            factor * (ecc + np.cos(true_anomaly)),
            np.zeros_like(radius),
        ),
        axis=-1,
    )
    rotation = perifocal_to_inertial(
        elements.raan_rad, elements.inclination_rad, elements.arg_perigee_rad
    )
    return (
        np.asarray(position_pf @ rotation.T, dtype=np.float64),
        np.asarray(velocity_pf @ rotation.T, dtype=np.float64),
    )


def state_transition_matrix(
    state: OrbitState, delta_time_s: float, gravitational_parameter: float = MU_EARTH
) -> Matrix:
    """Return the 6 by 6 state transition matrix from ``state.epoch_s`` to that epoch plus dt.

    The matrix is built by central differences of the analytic propagator. Steps
    are scaled to the position and velocity magnitudes so that the relative
    perturbation, and therefore the accuracy, is independent of the orbit size.
    """
    base = state.vector
    position_step = _STM_RELATIVE_STEP * float(np.linalg.norm(state.position_m))
    velocity_step = _STM_RELATIVE_STEP * float(np.linalg.norm(state.velocity_m_s))
    steps = np.array([position_step] * 3 + [velocity_step] * 3, dtype=np.float64)

    matrix = np.empty((6, 6), dtype=np.float64)
    for column in range(6):
        perturbation = np.zeros(6, dtype=np.float64)
        perturbation[column] = steps[column]
        forward = propagate(
            OrbitState.from_vector(base + perturbation, state.epoch_s),
            delta_time_s,
            gravitational_parameter,
        )
        backward = propagate(
            OrbitState.from_vector(base - perturbation, state.epoch_s),
            delta_time_s,
            gravitational_parameter,
        )
        matrix[:, column] = (forward.vector - backward.vector) / (2.0 * steps[column])
    return matrix


def symplectic_form() -> Matrix:
    """Return the 6 by 6 symplectic form ``J`` with blocks ``[[0, I], [-I, 0]]``."""
    form = np.zeros((6, 6), dtype=np.float64)
    form[:3, 3:] = np.eye(3)
    form[3:, :3] = -np.eye(3)
    return form


def symplectic_residual(transition: Matrix, time_scale_s: float) -> float:
    """Return the largest entry of ``Phi^T J Phi - J`` after non-dimensionalising ``Phi``.

    Two-body motion is the flow of a Hamiltonian system in the canonical pair
    (position, velocity) with unit mass, so its state transition matrix satisfies
    ``Phi^T J Phi = J`` exactly. Checking that identity on the raw matrix compares
    quantities with different units, because the position and velocity blocks of
    ``Phi`` carry factors of time. Conjugating by ``diag(I, T I)`` for a time scale
    ``T`` makes every block dimensionless and turns the residual into a number
    that can be compared against a fixed tolerance.

    Args:
        transition: The 6 by 6 state transition matrix.
        time_scale_s: A representative time for the orbit, for example the inverse
            of the mean motion.

    Returns:
        The largest absolute entry of the dimensionless residual.
    """
    if not time_scale_s > 0.0:
        raise ValueError("time_scale_s must be positive")
    scale = np.diag(np.array([1.0, 1.0, 1.0, time_scale_s, time_scale_s, time_scale_s]))
    inverse = np.diag(
        np.array([1.0, 1.0, 1.0, 1.0 / time_scale_s, 1.0 / time_scale_s, 1.0 / time_scale_s])
    )
    scaled = scale @ np.asarray(transition, dtype=np.float64) @ inverse
    form = symplectic_form()
    return float(np.max(np.abs(scaled.T @ form @ scaled - form)))


def propagate_covariance(covariance: Covariance, transition: Matrix) -> Covariance:
    """Map a 6 by 6 state covariance forward with ``C -> Phi C Phi^T``.

    This is the linear, or first order, propagation used by every operational
    conjunction assessment system. It is exact only in so far as the dynamics are
    linear over the propagation interval; see docs/design-notes.md for when that
    fails.
    """
    if covariance.dimension != 6:
        raise ValueError("state covariance propagation requires a 6 by 6 covariance")
    operator = np.asarray(transition, dtype=np.float64)
    if operator.shape != (6, 6):
        raise ValueError(f"transition matrix must be 6 by 6, got {operator.shape}")
    return Covariance(
        matrix=symmetrised(operator @ covariance.matrix @ operator.T), frame=covariance.frame
    )
