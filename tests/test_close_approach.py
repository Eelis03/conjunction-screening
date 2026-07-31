"""Property tests for close approach determination.

Two tolerances recur and both are derived from the search rather than from an
observed error.

The residual range rate is bounded by the curvature of the range multiplied by
the time tolerance of the root find. For a near-linear flyby with relative speed
``v`` and miss distance ``d`` the range obeys ``r(t)^2 = d^2 + v^2 t^2``, whose
range rate has slope ``v^2 / d`` at the closest approach. A root located to
within ``dt`` therefore leaves at most ``(v^2 / d) dt`` of range rate.

The agreement between the computed time of closest approach and the time the
encounter was constructed around is limited by the same root tolerance plus the
round trip error of the propagator. The second term is measured inside the test
by propagating the state forward and back and reading off the position it lost,
then dividing by the relative speed, because a position error ``dp`` displaces
the stationary point by ``dp / v``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjunction_screening.algorithm.close_approach import (
    CloseApproachSettings,
    coarse_step_for,
    find_close_approaches,
    relative_state,
)
from conjunction_screening.algorithm.filters import CascadeSettings, run_cascade
from conjunction_screening.algorithm.propagation import propagate
from conjunction_screening.model.state import OrbitState
from conjunction_screening.pipeline.catalog import (
    CatalogObject,
    PlantedConjunction,
    SyntheticCatalog,
)

_THRESHOLD_M = 5_000.0


def _search(
    catalog: SyntheticCatalog, secondary: CatalogObject, settings: CloseApproachSettings
) -> tuple[object, ...]:
    cascade = run_cascade(
        catalog.primary.state,
        secondary.state,
        CascadeSettings(threshold_m=_THRESHOLD_M, window_s=catalog.window_s),
    )
    assert cascade.passed
    return find_close_approaches(
        catalog.primary.state, secondary.state, cascade.candidate_windows, settings
    )


def _round_trip_position_error(state: OrbitState, delta_time_s: float) -> float:
    round_trip = propagate(propagate(state, delta_time_s), -delta_time_s)
    return float(np.linalg.norm(round_trip.position_m - state.position_m))


def test_time_of_closest_approach_has_zero_range_rate(
    conjunction_catalog: SyntheticCatalog, close_approach_settings: CloseApproachSettings
) -> None:
    """The defining condition of a closest approach holds at the returned time.

    Tolerance: ``2 (v^2 / d) dt`` with ``dt`` the root find tolerance, as derived
    in the module docstring. The factor of two absorbs the difference between the
    linear flyby model and the true curved relative motion.
    """
    for secondary in conjunction_catalog.secondaries:
        for approach in _search(conjunction_catalog, secondary, close_approach_settings):
            assert isinstance(approach.tca_s, float)
            assert approach.converged
            curvature = approach.relative_speed_m_s**2 / approach.miss_distance_m
            bound = 2.0 * curvature * close_approach_settings.time_tolerance_s
            assert abs(approach.range_rate_m_s) <= bound


def test_close_approach_matches_the_constructed_encounter(
    conjunction_catalog: SyntheticCatalog, close_approach_settings: CloseApproachSettings
) -> None:
    """The search recovers the time and distance the encounter was built around.

    Tolerance: the root find tolerance plus twice the propagator round trip
    position error divided by the relative speed, both measured rather than
    assumed. The miss distance tolerance is the same position error, because at a
    stationary point the distance is flat in time and the residual error is the
    propagation error itself.
    """
    for secondary, truth in zip(
        conjunction_catalog.secondaries, conjunction_catalog.planted, strict=True
    ):
        approaches = _search(conjunction_catalog, secondary, close_approach_settings)
        assert approaches
        best = min(approaches, key=lambda item: abs(item.tca_s - truth.tca_s))

        position_error = max(
            _round_trip_position_error(secondary.state, truth.tca_s),
            _round_trip_position_error(conjunction_catalog.primary.state, truth.tca_s),
        )
        time_tolerance = (
            close_approach_settings.time_tolerance_s
            + 2.0 * position_error / best.relative_speed_m_s
        )
        assert abs(best.tca_s - truth.tca_s) <= time_tolerance
        assert abs(best.miss_distance_m - truth.miss_distance_m) <= max(
            10.0 * position_error, 1e-6
        )
        assert best.relative_speed_m_s == pytest.approx(truth.relative_speed_m_s, rel=1e-9)


def test_miss_distance_is_symmetric_under_swapping_the_objects(
    conjunction_catalog: SyntheticCatalog, close_approach_settings: CloseApproachSettings
) -> None:
    """Which object is called the primary cannot change the answer.

    Tolerance: the two searches solve the same root of the same function, so they
    agree to the root find tolerance in time. The corresponding difference in
    miss distance is bounded by the relative speed multiplied by that tolerance,
    with a floor of a micrometre for rounding in the norm.
    """
    for secondary in conjunction_catalog.secondaries:
        cascade = run_cascade(
            conjunction_catalog.primary.state,
            secondary.state,
            CascadeSettings(threshold_m=_THRESHOLD_M, window_s=conjunction_catalog.window_s),
        )
        forward = find_close_approaches(
            conjunction_catalog.primary.state,
            secondary.state,
            cascade.candidate_windows,
            close_approach_settings,
        )
        reversed_order = find_close_approaches(
            secondary.state,
            conjunction_catalog.primary.state,
            cascade.candidate_windows,
            close_approach_settings,
        )
        assert len(forward) == len(reversed_order)
        for first, second in zip(forward, reversed_order, strict=True):
            assert abs(first.tca_s - second.tca_s) <= close_approach_settings.time_tolerance_s
            distance_tolerance = max(
                first.relative_speed_m_s * close_approach_settings.time_tolerance_s, 1e-6
            )
            assert abs(first.miss_distance_m - second.miss_distance_m) <= distance_tolerance
            assert np.allclose(
                first.relative_position_m, -np.asarray(second.relative_position_m), atol=1e-6
            )


def test_relative_state_is_antisymmetric() -> None:
    """Swapping the two objects negates the relative state exactly."""
    from conjunction_screening.pipeline.catalog import default_primary, generate_catalog

    catalog = generate_catalog(count=1, planted=1, seed=99)
    primary = default_primary().state
    secondary = catalog.secondaries[0].state
    times = np.linspace(0.0, 5_000.0, 17)
    forward_position, forward_velocity = relative_state(primary, secondary, times)
    reverse_position, reverse_velocity = relative_state(secondary, primary, times)
    assert np.array_equal(forward_position, -reverse_position)
    assert np.array_equal(forward_velocity, -reverse_velocity)


def test_coarse_step_resolves_the_threshold() -> None:
    """The coarse step is small enough that no approach reaching the threshold is skipped.

    The step is chosen so the relative motion over one step is a fraction of the
    threshold, so the sampled range cannot jump across the threshold band.
    """
    from conjunction_screening.pipeline.catalog import generate_catalog

    catalog = generate_catalog(count=2, planted=2, seed=1234)
    settings = CloseApproachSettings(threshold_m=_THRESHOLD_M)
    for secondary in catalog.secondaries:
        step = coarse_step_for(catalog.primary.state, secondary.state, settings)
        speed_bound = catalog.primary.state.speed_m_s + secondary.state.speed_m_s
        assert step * speed_bound <= settings.step_safety * _THRESHOLD_M + 1e-9


def test_approaches_outside_the_threshold_are_discarded(
    conjunction_catalog: SyntheticCatalog,
) -> None:
    """A tighter threshold returns a subset of the approaches a wider one returns."""
    secondary = conjunction_catalog.secondaries[0]
    wide = _search(
        conjunction_catalog, secondary, CloseApproachSettings(threshold_m=_THRESHOLD_M)
    )
    narrow = _search(conjunction_catalog, secondary, CloseApproachSettings(threshold_m=10.0))
    assert len(narrow) <= len(wide)
    for approach in narrow:
        assert approach.miss_distance_m <= 10.0


def test_settings_reject_invalid_values() -> None:
    """Configuration errors are refused at construction."""
    with pytest.raises(ValueError, match="step_safety"):
        CloseApproachSettings(step_safety=0.0)
    with pytest.raises(ValueError, match="time_tolerance_s"):
        CloseApproachSettings(time_tolerance_s=0.0)


def test_planted_truth_records_a_stationary_point(
    conjunction_catalog: SyntheticCatalog,
) -> None:
    """The construction really does place a stationary point at the recorded time.

    Without this the previous tests would be comparing against an arbitrary
    label rather than against ground truth.
    """
    for secondary, truth in zip(
        conjunction_catalog.secondaries, conjunction_catalog.planted, strict=True
    ):
        assert isinstance(truth, PlantedConjunction)
        offsets, rates = relative_state(
            conjunction_catalog.primary.state,
            secondary.state,
            np.array([truth.tca_s], dtype=np.float64),
        )
        distance = float(np.linalg.norm(offsets[0]))
        range_rate = float(np.dot(offsets[0], rates[0])) / distance
        curvature = truth.relative_speed_m_s**2 / truth.miss_distance_m
        position_error = _round_trip_position_error(secondary.state, truth.tca_s)
        assert abs(range_rate) <= curvature * (position_error / truth.relative_speed_m_s) + 1e-3
        assert distance == pytest.approx(
            truth.miss_distance_m, abs=max(10.0 * position_error, 1e-6)
        )
