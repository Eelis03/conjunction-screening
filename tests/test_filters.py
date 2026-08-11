"""Property tests for the filter cascade.

Two properties matter and they pull in opposite directions.

Conservativeness is the safety property. A filter may reject a pair only when no
separation below the threshold is possible. It is tested by generating pairs that
demonstrably do conjunct, from the planted construction whose time of closest
approach and miss distance are known by construction, and asserting that no
filter rejects any of them. A filter that always passed would satisfy this, which
is why the second property is needed.

Selectivity is the usefulness property. Each filter must reject pairs that
provably cannot conjunct. Those pairs are built by hand from geometry that can be
checked on paper, one for each filter, so that the test pins the filter it names
rather than passing because an earlier filter happened to fire.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjunction_screening.algorithm.filters import (
    PATH_FILTER,
    PERIGEE_APOGEE_FILTER,
    TIME_FILTER,
    CascadeSettings,
    at_risk_arcs,
    minimum_path_separation,
    orbit_path_filter,
    perigee_apogee_filter,
    run_cascade,
    time_filter,
)
from conjunction_screening.algorithm.propagation import propagate
from conjunction_screening.model.constants import EARTH_RADIUS_M, MU_EARTH
from conjunction_screening.model.state import KeplerianElements, elements_from_state
from conjunction_screening.pipeline.catalog import SyntheticCatalog, generate_catalog

_THRESHOLD_M = 5_000.0
_TWO_PI = 2.0 * np.pi


def _covered(arcs: tuple[tuple[float, float], ...], anomaly: float) -> bool:
    return any(
        begin <= anomaly + _TWO_PI * turn <= end for begin, end in arcs for turn in (-1, 0, 1)
    )


@pytest.mark.parametrize("seed", [11, 2029, 4021])
def test_no_filter_rejects_a_real_conjunction(seed: int) -> None:
    """The safety property: a pair that does conjunct survives every filter.

    Each secondary in this catalogue was constructed so that at a known time its
    separation from the primary equals a known value below the threshold. Any
    rejection is therefore a proved defect rather than a tolerance question, and
    the assertion needs no tolerance at all.
    """
    catalog = generate_catalog(count=10, planted=10, window_s=86_400.0, seed=seed)
    primary = elements_from_state(catalog.primary.state)
    settings = CascadeSettings(threshold_m=_THRESHOLD_M, window_s=86_400.0)

    for secondary, truth in zip(catalog.secondaries, catalog.planted, strict=True):
        assert truth.miss_distance_m < _THRESHOLD_M
        elements = elements_from_state(secondary.state)

        first = perigee_apogee_filter(primary, elements, _THRESHOLD_M)
        assert first.passed, f"{truth.object_id}: {first.detail}"

        second = orbit_path_filter(primary, elements, _THRESHOLD_M, settings)
        assert second.passed, f"{truth.object_id}: {second.detail}"

        third, windows = time_filter(primary, elements, _THRESHOLD_M, settings)
        assert third.passed, f"{truth.object_id}: {third.detail}"
        assert any(begin <= truth.tca_s <= end for begin, end in windows), (
            f"{truth.object_id}: the true time of closest approach is outside every window"
        )


def test_cascade_passes_every_planted_conjunction(conjunction_catalog: SyntheticCatalog) -> None:
    """The whole cascade agrees with the individual filters on the safety property."""
    settings = CascadeSettings(threshold_m=_THRESHOLD_M, window_s=86_400.0)
    for secondary, truth in zip(
        conjunction_catalog.secondaries, conjunction_catalog.planted, strict=True
    ):
        result = run_cascade(conjunction_catalog.primary.state, secondary.state, settings)
        assert result.passed, f"{truth.object_id} rejected by {result.rejected_by}"
        assert result.rejected_by is None
        assert len(result.verdicts) == 3
        assert result.candidate_windows


@pytest.mark.parametrize("seed", [77, 4021])
def test_at_risk_arcs_contain_the_true_conjunction_anomalies(seed: int) -> None:
    """The arcs handed to the time filter cover the true anomalies of a real conjunction.

    This is the inequality the time filter depends on. If an arc missed the true
    anomaly at which the encounter happens, the filter could conclude that the
    two objects are never simultaneously at risk when in fact they are.
    """
    catalog = generate_catalog(count=8, planted=8, window_s=86_400.0, seed=seed)
    primary = elements_from_state(catalog.primary.state)
    for secondary, truth in zip(catalog.secondaries, catalog.planted, strict=True):
        elements = elements_from_state(secondary.state)
        primary_arcs, secondary_arcs = at_risk_arcs(primary, elements, _THRESHOLD_M)
        primary_at_tca = elements_from_state(propagate(catalog.primary.state, truth.tca_s))
        secondary_at_tca = elements_from_state(propagate(secondary.state, truth.tca_s))
        assert _covered(primary_arcs, primary_at_tca.true_anomaly_rad), truth.object_id
        assert _covered(secondary_arcs, secondary_at_tca.true_anomaly_rad), truth.object_id


def test_perigee_apogee_filter_rejects_disjoint_shells() -> None:
    """Two circular orbits eight hundred kilometres apart cannot conjunct.

    The two radial shells are single radii, so the gap is exact and no tolerance
    enters the assertion.
    """
    low = KeplerianElements(EARTH_RADIUS_M + 400e3, 0.0, 0.5, 0.0, 0.0, 0.0, MU_EARTH)
    high = KeplerianElements(EARTH_RADIUS_M + 1_200e3, 0.0, 1.9, 2.0, 0.0, 3.0, MU_EARTH)
    verdict = perigee_apogee_filter(low, high, _THRESHOLD_M)
    assert not verdict.passed
    assert verdict.name == PERIGEE_APOGEE_FILTER
    assert verdict.bound == pytest.approx(800e3, rel=1e-12)


def test_perigee_apogee_filter_passes_overlapping_shells() -> None:
    """Overlapping shells cannot be ruled out by radius alone."""
    circular = KeplerianElements(7.0e6, 0.0, 0.5, 0.0, 0.0, 0.0, MU_EARTH)
    eccentric = KeplerianElements(7.0e6, 0.05, 1.9, 2.0, 0.0, 3.0, MU_EARTH)
    verdict = perigee_apogee_filter(circular, eccentric, _THRESHOLD_M)
    assert verdict.passed
    assert verdict.bound < 0.0


def _crossing_pair() -> tuple[KeplerianElements, KeplerianElements]:
    """Return a coplanar-shell pair whose paths stay seventeen kilometres apart.

    Orbit one is a circle of radius 7000 km in the reference plane. Orbit two has
    the same semi-major axis, an eccentricity of 0.05, an inclination of ninety
    degrees, and its perigee ninety degrees from the ascending node. It therefore
    crosses the reference plane at true anomalies of plus and minus ninety
    degrees, where its radius equals the semi-latus rectum, 6982.5 km. Every
    other point of orbit two lies out of the plane and is further from the circle,
    so the minimum separation between the two paths is 17.5 km. The radial shells
    overlap by a wide margin, so the perigee and apogee filter cannot fire and the
    orbit path filter is the one under test.
    """
    circle = KeplerianElements(7.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, MU_EARTH)
    ellipse = KeplerianElements(7.0e6, 0.05, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.0, MU_EARTH)
    return circle, ellipse


def test_orbit_path_filter_rejects_paths_that_never_meet() -> None:
    """The orbit path filter fires on a pair the radial filter cannot separate."""
    circle, ellipse = _crossing_pair()
    assert perigee_apogee_filter(circle, ellipse, _THRESHOLD_M).passed

    verdict = orbit_path_filter(circle, ellipse, _THRESHOLD_M)
    assert not verdict.passed
    assert verdict.name == PATH_FILTER
    assert verdict.bound > _THRESHOLD_M


def test_orbit_path_bound_brackets_the_true_minimum() -> None:
    """The reported lower bound never exceeds the analytically known minimum.

    The minimum separation of the constructed pair is exactly
    ``7000 km - 7000 km * (1 - 0.05^2)``, which is 17.5 km. The branch and bound
    must report a lower bound at or below that value, otherwise it would be
    claiming more than it has proved.
    """
    circle, ellipse = _crossing_pair()
    exact = 7.0e6 - 7.0e6 * (1.0 - 0.05**2)
    separation = minimum_path_separation(circle, ellipse, _THRESHOLD_M)
    assert not separation.can_approach
    assert separation.lower_bound_m <= exact
    assert separation.witness_m >= exact
    assert separation.cells_evaluated > 0


def test_orbit_path_filter_passes_intersecting_paths() -> None:
    """Two circles of equal radius in different planes intersect and cannot be ruled out."""
    first = KeplerianElements(7.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, MU_EARTH)
    second = KeplerianElements(7.0e6, 0.0, np.deg2rad(60.0), 0.0, 0.0, 0.0, MU_EARTH)
    verdict = orbit_path_filter(first, second, _THRESHOLD_M)
    assert verdict.passed
    assert verdict.bound <= _THRESHOLD_M


def test_time_filter_rejects_paths_that_cross_out_of_phase() -> None:
    """Equal periods with a fixed quarter-period offset never bring the objects together.

    Both orbits are circles of radius 7000 km, so they share a period exactly.
    Their planes meet along the reference direction, so the paths intersect at two
    points. Object one starts at one crossing point and object two starts a
    quarter of a revolution before its own. Because the periods are equal that
    offset never changes, so the two are never at a crossing point at the same
    time. The orbit path filter cannot fire, because the paths do intersect.
    """
    settings = CascadeSettings(threshold_m=_THRESHOLD_M, window_s=86_400.0)
    first = KeplerianElements(7.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, MU_EARTH)
    second = KeplerianElements(7.0e6, 0.0, np.deg2rad(60.0), 0.0, 0.0, 0.5 * np.pi, MU_EARTH)

    assert perigee_apogee_filter(first, second, _THRESHOLD_M).passed
    assert orbit_path_filter(first, second, _THRESHOLD_M, settings).passed

    verdict, windows = time_filter(first, second, _THRESHOLD_M, settings)
    assert not verdict.passed
    assert verdict.name == TIME_FILTER
    assert windows == ()
    assert verdict.bound > 0.0


def test_time_filter_passes_when_the_phasing_lines_up() -> None:
    """The same geometry with both objects at the crossing point at once passes."""
    settings = CascadeSettings(threshold_m=_THRESHOLD_M, window_s=86_400.0)
    first = KeplerianElements(7.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, MU_EARTH)
    second = KeplerianElements(7.0e6, 0.0, np.deg2rad(60.0), 0.0, 0.0, 0.0, MU_EARTH)
    verdict, windows = time_filter(first, second, _THRESHOLD_M, settings)
    assert verdict.passed
    assert windows


def test_cascade_short_circuits_on_the_first_rejection() -> None:
    """A pair rejected by the cheapest filter never reaches the expensive ones."""
    settings = CascadeSettings(threshold_m=_THRESHOLD_M, window_s=86_400.0)
    low = KeplerianElements(EARTH_RADIUS_M + 400e3, 0.0, 0.5, 0.0, 0.0, 0.0, MU_EARTH)
    high = KeplerianElements(EARTH_RADIUS_M + 1_200e3, 0.0, 1.9, 2.0, 0.0, 3.0, MU_EARTH)
    from conjunction_screening.model.state import state_from_elements

    result = run_cascade(state_from_elements(low), state_from_elements(high), settings)
    assert not result.passed
    assert result.rejected_by == PERIGEE_APOGEE_FILTER
    assert len(result.verdicts) == 1
    assert result.candidate_windows == ()


def test_a_wider_threshold_never_rejects_more(mixed_catalog: SyntheticCatalog) -> None:
    """Raising the threshold can only make the cascade more permissive.

    Monotonicity in the threshold is a structural property of a conservative
    cascade: every rejection is an inequality against the threshold, so widening
    it can only turn rejections into passes.
    """
    tight = CascadeSettings(threshold_m=2_000.0, window_s=86_400.0)
    wide = CascadeSettings(threshold_m=20_000.0, window_s=86_400.0)
    for secondary in mixed_catalog.secondaries[:25]:
        narrow = run_cascade(mixed_catalog.primary.state, secondary.state, tight)
        broad = run_cascade(mixed_catalog.primary.state, secondary.state, wide)
        if narrow.passed:
            assert broad.passed, secondary.object_id


def test_cascade_settings_reject_invalid_values() -> None:
    """Configuration errors are refused at construction."""
    with pytest.raises(ValueError, match="threshold_m"):
        CascadeSettings(threshold_m=0.0)
    with pytest.raises(ValueError, match="window_s"):
        CascadeSettings(window_s=-1.0)
    with pytest.raises(ValueError, match="path_divisions"):
        CascadeSettings(path_divisions=2)
