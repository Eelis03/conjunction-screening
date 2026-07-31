"""Property tests for propagation and covariance propagation.

The state transition matrix is built by central differences with a relative step
of 1e-6, so each entry carries a relative error of roughly 1e-10. The symplectic
identity is a quadratic form in that matrix, so the residual it leaves scales
with the square of the largest entry. Every tolerance on the symplectic residual
below is written that way rather than as a fixed number, because the largest
entry grows linearly with the propagation interval.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjunction_screening.algorithm.propagation import (
    propagate,
    propagate_covariance,
    propagate_many,
    propagate_to,
    state_transition_matrix,
    symplectic_residual,
)
from conjunction_screening.model.constants import EARTH_RADIUS_M, MU_EARTH
from conjunction_screening.model.covariance import (
    Covariance,
    ElementSigmas,
    covariance_from_element_sigmas,
    is_symmetric_positive_semidefinite,
)
from conjunction_screening.model.state import (
    KeplerianElements,
    OrbitState,
    elements_from_state,
    state_from_elements,
)

_STM_RELATIVE_ERROR = 1e-8
"""Bound on the relative error of one central-difference state transition entry.

Central differencing balances a truncation error of order ``h^2`` against a
round-off error of order ``eps / h``. At the relative step of 1e-6 the propagator
uses, those are 1e-12 and 2e-10. The value here is two orders above that sum, so
the tests are checking the method rather than a particular rounding.
"""


def _reference_state() -> OrbitState:
    elements = KeplerianElements(
        semi_major_axis_m=EARTH_RADIUS_M + 700e3,
        eccentricity=1.2e-3,
        inclination_rad=1.714,
        raan_rad=0.419,
        arg_perigee_rad=0.908,
        true_anomaly_rad=0.192,
        gravitational_parameter=MU_EARTH,
    )
    return state_from_elements(elements, epoch_s=0.0)


def _scaled_transition(state: OrbitState, delta_time_s: float) -> tuple[np.ndarray, float]:
    elements = elements_from_state(state)
    time_scale = 1.0 / elements.mean_motion_rad_s
    transition = state_transition_matrix(state, delta_time_s)
    scale = np.diag([1.0, 1.0, 1.0, time_scale, time_scale, time_scale])
    inverse = np.diag(
        [1.0, 1.0, 1.0, 1.0 / time_scale, 1.0 / time_scale, 1.0 / time_scale]
    )
    return scale @ transition @ inverse, time_scale


@pytest.mark.parametrize("delta_time_s", [60.0, 1_800.0, 21_600.0, 86_400.0])
def test_state_transition_matrix_is_symplectic(delta_time_s: float) -> None:
    """Two-body motion is Hamiltonian, so its transition matrix satisfies ``Phi^T J Phi = J``.

    Tolerance: the residual is a product of two transition matrices, so an entry
    error of relative size ``_STM_RELATIVE_ERROR`` produces a residual of order
    that value multiplied by the square of the largest non-dimensional entry.
    The bound below is written in exactly those terms, which is why it can be
    applied unchanged from a one minute to a one day propagation.
    """
    state = _reference_state()
    scaled, time_scale = _scaled_transition(state, delta_time_s)
    largest = float(np.max(np.abs(scaled)))
    residual = symplectic_residual(state_transition_matrix(state, delta_time_s), time_scale)
    assert residual <= _STM_RELATIVE_ERROR * largest**2


def test_zero_interval_transition_is_the_identity() -> None:
    """Propagating by no time at all moves nothing.

    The comparison is made on the non-dimensional matrix, because the raw matrix
    mixes blocks measured in seconds with blocks measured in reciprocal seconds
    and a single absolute tolerance would mean four different things.

    Tolerance: a zero-interval propagation still passes through the element
    conversion, which loses about 1e-15 relative on a position of order 1e7 m.
    Dividing that by the finite-difference step, which is 1e-6 relative, leaves
    about 1e-9 in every non-dimensional entry. The bound of 1e-7 keeps two orders
    of margin.
    """
    scaled, _ = _scaled_transition(_reference_state(), 0.0)
    assert np.allclose(scaled, np.eye(6), atol=1e-7)


@pytest.mark.parametrize("delta_time_s", [123.0, 5_000.0, 43_200.0])
def test_propagation_conserves_energy_and_angular_momentum(delta_time_s: float) -> None:
    """Analytic two-body propagation conserves both integrals of the motion.

    Tolerance: propagation converts to elements, advances the anomaly, and
    converts back, which is a bounded sequence of operations on values of order
    1e7, so the relative drift is a small multiple of machine epsilon.
    """
    state = _reference_state()
    moved = propagate(state, delta_time_s)

    def energy(item: OrbitState) -> float:
        return 0.5 * item.speed_m_s**2 - MU_EARTH / item.radius_m

    assert energy(moved) == pytest.approx(energy(state), rel=1e-12)
    assert np.allclose(
        np.cross(moved.position_m, moved.velocity_m_s),
        np.cross(state.position_m, state.velocity_m_s),
        rtol=1e-12,
    )


@pytest.mark.parametrize("delta_time_s", [600.0, 20_000.0, 86_400.0])
def test_propagation_is_reversible(delta_time_s: float) -> None:
    """Propagating forward and back recovers the state to well below a millimetre.

    Tolerance: the round trip inherits the relative error of two element
    conversions and one Kepler solve, which is a few times machine epsilon on a
    position of order 1e7 m, so a micrometre is a generous bound.
    """
    state = _reference_state()
    round_trip = propagate(propagate(state, delta_time_s), -delta_time_s)
    assert float(np.linalg.norm(round_trip.position_m - state.position_m)) < 1e-6
    assert float(np.linalg.norm(round_trip.velocity_m_s - state.velocity_m_s)) < 1e-9


def test_propagate_many_matches_repeated_single_propagation() -> None:
    """The vectorised propagator agrees with the scalar one it replaces."""
    state = _reference_state()
    offsets = np.linspace(0.0, 6_000.0, 25)
    positions, velocities = propagate_many(state, offsets)
    for index, offset in enumerate(offsets):
        single = propagate(state, float(offset))
        assert np.allclose(positions[index], single.position_m, rtol=1e-12, atol=1e-6)
        assert np.allclose(velocities[index], single.velocity_m_s, rtol=1e-12, atol=1e-9)


def test_propagate_to_uses_absolute_epochs() -> None:
    """Propagating to an epoch and by an interval agree when the two match."""
    state = _reference_state()
    by_interval = propagate(state, 1_234.0)
    to_epoch = propagate_to(state, 1_234.0)
    assert np.allclose(by_interval.position_m, to_epoch.position_m, rtol=1e-14)
    assert to_epoch.epoch_s == pytest.approx(1_234.0)


@pytest.mark.parametrize("delta_time_s", [0.0, 3_600.0, 43_200.0, 86_400.0])
def test_propagated_covariance_stays_symmetric_positive_semidefinite(
    delta_time_s: float,
) -> None:
    """Every stage of covariance propagation preserves the defining properties."""
    state = _reference_state()
    elements = elements_from_state(state)
    sigmas = ElementSigmas(15.0, 2.0e-6, 4.0e-5, 4.0e-5, 6.0e-5, 6.0e-5)
    covariance = covariance_from_element_sigmas(elements, sigmas)
    assert is_symmetric_positive_semidefinite(covariance)

    transition = state_transition_matrix(state, delta_time_s)
    propagated = propagate_covariance(covariance, transition)
    assert is_symmetric_positive_semidefinite(propagated)
    assert is_symmetric_positive_semidefinite(propagated.position_block())


def test_in_track_uncertainty_grows_with_semi_major_axis_uncertainty() -> None:
    """A semi-major axis error drives in-track growth; a well-known axis does not.

    This is the reason the synthetic catalogue builds its covariances in element
    space. The test compares two objects that differ only in their semi-major
    axis uncertainty and checks that the looser one grows a great deal more over
    one day.
    """
    state = _reference_state()
    elements = elements_from_state(state)
    transition = state_transition_matrix(state, 86_400.0)

    def in_track_sigma(semi_major_sigma_m: float) -> float:
        sigmas = ElementSigmas(semi_major_sigma_m, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9)
        propagated = propagate_covariance(
            covariance_from_element_sigmas(elements, sigmas), transition
        )
        return float(np.sqrt(np.trace(propagated.position_block().matrix)))

    tight = in_track_sigma(1.0)
    loose = in_track_sigma(50.0)
    assert loose > 40.0 * tight


def test_propagate_covariance_rejects_wrong_dimension() -> None:
    """A position-only covariance cannot be propagated with a state transition matrix."""
    with pytest.raises(ValueError, match="6 by 6"):
        propagate_covariance(Covariance(matrix=np.eye(3)), np.eye(6))
