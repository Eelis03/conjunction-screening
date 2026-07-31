"""Property tests for the model layer.

Tolerances here come from one of two places. Conversions between states and
elements are a few dozen arithmetic operations on quantities of order 1e7, so
their relative error is a small multiple of machine epsilon; the tolerances are
set several orders of magnitude above that. Kepler's equation is solved to a
stated Newton step tolerance, and the residual it leaves is bounded by that step
multiplied by the derivative of the equation, which is at most two.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjunction_screening.model.arrays import as_vector, symmetrised, unit_vector
from conjunction_screening.model.constants import EARTH_RADIUS_M, MU_EARTH
from conjunction_screening.model.covariance import (
    Covariance,
    ElementSigmas,
    combine_covariances,
    covariance_from_element_sigmas,
    is_symmetric_positive_semidefinite,
)
from conjunction_screening.model.encounter import (
    encounter_plane_basis,
    planar_encounter,
    principal_axis_form,
    project_to_encounter_plane,
)
from conjunction_screening.model.frames import (
    inertial_to_ric_rotation,
    rotate_covariance_to_inertial,
    rotate_covariance_to_ric,
)
from conjunction_screening.model.state import (
    KeplerianElements,
    OrbitState,
    eccentric_to_mean_anomaly,
    elements_from_state,
    path_positions,
    solve_kepler_equation,
    state_from_elements,
    true_to_eccentric_anomaly,
)

_ELEMENT_CASES = [
    (EARTH_RADIUS_M + 700e3, 1.2e-3, 1.71, 0.42, 0.91, 0.19),
    (EARTH_RADIUS_M + 400e3, 1.0e-4, 0.90, 2.10, 4.50, 3.30),
    (EARTH_RADIUS_M + 1_200e3, 0.05, 2.60, 5.90, 1.20, 5.10),
    (EARTH_RADIUS_M + 900e3, 0.30, 1.00, 0.10, 3.00, 0.60),
]


def _elements(case: tuple[float, float, float, float, float, float]) -> KeplerianElements:
    return KeplerianElements(
        semi_major_axis_m=case[0],
        eccentricity=case[1],
        inclination_rad=case[2],
        raan_rad=case[3],
        arg_perigee_rad=case[4],
        true_anomaly_rad=case[5],
        gravitational_parameter=MU_EARTH,
    )


@pytest.mark.parametrize("case", _ELEMENT_CASES)
def test_element_state_round_trip(case: tuple[float, float, float, float, float, float]) -> None:
    """Converting elements to a state and back reproduces the elements.

    Tolerance: the conversion is a bounded sequence of arithmetic operations on
    values of order 1e7, so its relative error is a few times machine epsilon.
    A relative tolerance of 1e-10 leaves five orders of margin.
    """
    original = _elements(case)
    recovered = elements_from_state(state_from_elements(original))
    assert recovered.semi_major_axis_m == pytest.approx(original.semi_major_axis_m, rel=1e-10)
    assert recovered.eccentricity == pytest.approx(original.eccentricity, rel=1e-8, abs=1e-12)
    for name in ("inclination_rad", "raan_rad", "arg_perigee_rad", "true_anomaly_rad"):
        assert getattr(recovered, name) == pytest.approx(getattr(original, name), abs=1e-9)


@pytest.mark.parametrize("eccentricity", [0.0, 1e-4, 0.05, 0.3, 0.7])
def test_kepler_equation_residual(eccentricity: float) -> None:
    """The solved eccentric anomaly satisfies Kepler's equation.

    Tolerance: the solver stops when its Newton step falls below 1e-13 rad, and
    the residual left by a step of that size is at most the step multiplied by
    the derivative ``1 - e cos E``, which never exceeds two. The bound is
    therefore 2e-13; 1e-12 is used.
    """
    mean = np.linspace(-8.0 * np.pi, 8.0 * np.pi, 501)
    ecc_anomaly = solve_kepler_equation(mean, eccentricity)
    wrapped = np.mod(mean + np.pi, 2.0 * np.pi) - np.pi
    residual = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - wrapped
    assert float(np.max(np.abs(residual))) < 1e-12


@pytest.mark.parametrize("eccentricity", [0.0, 1e-3, 0.05, 0.3, 0.7, 0.9])
def test_path_speed_bound_is_an_upper_bound(eccentricity: float) -> None:
    """The Lipschitz constant used by the orbit path filter really bounds the path speed.

    This is the inequality the whole orbit path filter rests on. If the bound
    were ever exceeded, the filter could prune a cell that contains a real
    conjunction. The check compares the closed-form bound against a finely
    sampled numerical derivative; the chord between samples underestimates the
    derivative, so an exceedance would be a genuine failure rather than a
    sampling artefact.
    """
    elements = KeplerianElements(7.0e6, eccentricity, 0.5, 0.3, 0.2, 0.0)
    anomalies = np.linspace(0.0, 2.0 * np.pi, 200_001)
    positions = path_positions(elements, anomalies)
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(anomalies)
    assert float(np.max(speeds)) <= elements.path_speed_bound_m_rad


def test_orbit_state_arrays_are_read_only() -> None:
    """A frozen state cannot be mutated through a reference to one of its arrays."""
    state = state_from_elements(_elements(_ELEMENT_CASES[0]))
    with pytest.raises(ValueError, match="read-only"):
        state.position_m[0] = 0.0


def test_as_vector_rejects_wrong_shape() -> None:
    """Shape checking happens on construction rather than at first use."""
    with pytest.raises(ValueError, match="shape"):
        as_vector([1.0, 2.0], 3, "sample")


def test_unit_vector_rejects_zero() -> None:
    """A zero vector has no direction and is a modelling error."""
    with pytest.raises(ValueError, match="no direction"):
        unit_vector(np.zeros(3), "sample")


@pytest.mark.parametrize("case", _ELEMENT_CASES)
def test_ric_rotation_is_orthonormal(
    case: tuple[float, float, float, float, float, float],
) -> None:
    """The RIC axes form a right-handed orthonormal triad.

    Tolerance: the rotation is built from two normalisations and one cross
    product, so its departure from orthonormality is a few times machine epsilon.
    """
    state = state_from_elements(_elements(case))
    rotation = inertial_to_ric_rotation(state)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("case", _ELEMENT_CASES)
def test_covariance_frame_round_trip(
    case: tuple[float, float, float, float, float, float],
) -> None:
    """Rotating a covariance into RIC and back reproduces it and its eigenvalues."""
    elements = _elements(case)
    state = state_from_elements(elements)
    sigmas = ElementSigmas(12.0, 2.0e-6, 3.0e-5, 4.0e-5, 5.0e-5, 6.0e-5)
    inertial = covariance_from_element_sigmas(elements, sigmas)
    ric = rotate_covariance_to_ric(inertial, state)
    back = rotate_covariance_to_inertial(ric, state)
    assert np.allclose(back.matrix, inertial.matrix, rtol=1e-12, atol=0.0)
    assert np.allclose(
        np.linalg.eigvalsh(ric.matrix), np.linalg.eigvalsh(inertial.matrix), rtol=1e-10
    )
    assert is_symmetric_positive_semidefinite(ric)


@pytest.mark.parametrize("case", _ELEMENT_CASES)
def test_element_covariance_is_positive_semidefinite(
    case: tuple[float, float, float, float, float, float],
) -> None:
    """A covariance built as ``J diag(sigma^2) J^T`` is semi-definite by construction."""
    elements = _elements(case)
    sigmas = ElementSigmas(20.0, 3.0e-6, 5.0e-5, 5.0e-5, 8.0e-5, 8.0e-5)
    covariance = covariance_from_element_sigmas(elements, sigmas)
    assert covariance.dimension == 6
    assert is_symmetric_positive_semidefinite(covariance)


def test_covariance_rejects_asymmetric_input() -> None:
    """A matrix that is not symmetric is not a covariance."""
    with pytest.raises(ValueError, match="not symmetric"):
        Covariance(matrix=np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_combine_covariances_requires_matching_frames() -> None:
    """Summing covariances quoted in different frames is a units error."""
    first = Covariance(matrix=np.eye(3), frame="ECI")
    second = Covariance(matrix=np.eye(3), frame="RIC")
    with pytest.raises(ValueError, match="share a frame"):
        combine_covariances(first, second)


def test_symmetrised_removes_rounding_asymmetry() -> None:
    """Averaging a matrix with its transpose leaves symmetric matrices unchanged."""
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert np.array_equal(symmetrised(matrix), matrix)


def test_encounter_basis_is_orthonormal_and_normal_to_relative_velocity() -> None:
    """The encounter plane basis spans the plane perpendicular to the relative velocity."""
    position = np.array([120.0, -340.0, 55.0])
    velocity = np.array([-4_000.0, 900.0, 6_100.0])
    basis = encounter_plane_basis(position, velocity)
    assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-14)
    assert np.allclose(basis @ velocity, np.zeros(2), atol=1e-9 * float(np.linalg.norm(velocity)))


def test_encounter_projection_preserves_perpendicular_magnitude() -> None:
    """Projecting a relative position that is already perpendicular preserves its length.

    Tolerance: the relative position used here is exactly perpendicular to the
    relative velocity, so the projection is an isometry on it up to rounding in
    the two dot products, which is a few times machine epsilon relative.
    """
    velocity = np.array([0.0, 0.0, 7_500.0])
    position = np.array([310.0, -180.0, 0.0])
    basis = encounter_plane_basis(position, velocity)
    projected = project_to_encounter_plane(position, basis)
    assert float(np.linalg.norm(projected)) == pytest.approx(
        float(np.linalg.norm(position)), rel=1e-14
    )


def test_encounter_projection_loses_only_the_along_track_component() -> None:
    """A relative position with an along-velocity part loses exactly that part."""
    velocity = np.array([0.0, 0.0, 7_500.0])
    position = np.array([310.0, -180.0, 640.0])
    basis = encounter_plane_basis(position, velocity)
    projected = project_to_encounter_plane(position, basis)
    expected = float(np.hypot(310.0, 180.0))
    assert float(np.linalg.norm(projected)) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("orientation", [0.0, 0.3, 1.1, 2.5])
def test_principal_axis_form_preserves_miss_and_determinant(orientation: float) -> None:
    """Diagonalising the in-plane covariance is a rotation, so it preserves both invariants."""
    encounter = planar_encounter(
        miss_distance_m=420.0,
        sigma_x_m=1_500.0,
        sigma_y_m=180.0,
        hard_body_radius_m=11.0,
        orientation_rad=orientation,
    )
    form = principal_axis_form(encounter)
    assert float(np.hypot(form.mean_x_m, form.mean_y_m)) == pytest.approx(420.0, rel=1e-12)
    determinant = float(np.linalg.det(encounter.plane_covariance.matrix))
    assert (form.sigma_x_m * form.sigma_y_m) ** 2 == pytest.approx(determinant, rel=1e-10)
    assert form.sigma_x_m == pytest.approx(1_500.0, rel=1e-12)
    assert form.sigma_y_m == pytest.approx(180.0, rel=1e-12)


def test_planar_encounter_reports_the_requested_geometry() -> None:
    """The constructor round trips through the encounter representation."""
    encounter = planar_encounter(
        miss_distance_m=250.0, sigma_x_m=600.0, sigma_y_m=90.0, hard_body_radius_m=9.0
    )
    assert encounter.projected_miss_distance_m == pytest.approx(250.0, rel=1e-14)
    assert encounter.miss_distance_m == pytest.approx(250.0, rel=1e-14)
    assert encounter.hard_body_radius_m == 9.0
    assert is_symmetric_positive_semidefinite(encounter.plane_covariance)
    assert is_symmetric_positive_semidefinite(encounter.relative_covariance)


def test_mean_anomaly_conversions_are_inverse() -> None:
    """True anomaly to eccentric to mean and back is the identity."""
    eccentricity = 0.12
    true_anomaly = np.linspace(0.01, 2.0 * np.pi - 0.01, 97)
    ecc_anomaly = true_to_eccentric_anomaly(true_anomaly, eccentricity)
    mean = eccentric_to_mean_anomaly(ecc_anomaly, eccentricity)
    recovered = solve_kepler_equation(mean, eccentricity)
    assert np.allclose(np.mod(recovered, 2.0 * np.pi), np.mod(ecc_anomaly, 2.0 * np.pi), atol=1e-11)


def test_open_orbit_is_rejected() -> None:
    """A hyperbolic state is outside the domain of this library and says so."""
    state = OrbitState(
        epoch_s=0.0,
        position_m=np.array([7.0e6, 0.0, 0.0]),
        velocity_m_s=np.array([0.0, 12_000.0, 0.0]),
    )
    with pytest.raises(ValueError, match="open orbit"):
        elements_from_state(state)
