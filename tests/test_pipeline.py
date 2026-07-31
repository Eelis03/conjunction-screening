"""Property tests for the pipeline and analysis layers.

The covariance chain runs RIC at the object epoch, to the inertial frame, forward
with the state transition matrix, down to the position block, summed with the
other object, back to RIC at the time of closest approach, and finally projected
into the encounter plane. Every stage is a congruence transform, so symmetry and
positive semi-definiteness must survive all of them. That is the invariant this
module checks at every stage rather than only at the end.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjunction_screening.algorithm.close_approach import CloseApproachSettings
from conjunction_screening.algorithm.probability import (
    ALFANO,
    CHAN,
    FOSTER,
    AlfanoMethod,
    ChanMethod,
    FosterMethod,
)
from conjunction_screening.analysis.comparison import (
    compare_methods,
    format_comparison_table,
    relative_disagreement,
    worst_pairwise_disagreement,
)
from conjunction_screening.analysis.ranking import (
    ActionClass,
    ActionThresholds,
    format_ranking_table,
    rank_events,
    rank_report,
)
from conjunction_screening.model.covariance import is_symmetric_positive_semidefinite
from conjunction_screening.pipeline.catalog import (
    SyntheticCatalog,
    default_primary,
    generate_catalog,
)
from conjunction_screening.pipeline.screening import (
    ScreeningConfig,
    ScreeningReport,
    build_encounter,
    propagated_position_covariance,
    run_screening,
)


def test_catalog_generation_is_reproducible_from_its_seed() -> None:
    """Two catalogues from the same seed are identical."""
    first = generate_catalog(count=15, planted=3, seed=606)
    second = generate_catalog(count=15, planted=3, seed=606)
    assert first.size == second.size == 15
    for left, right in zip(first.secondaries, second.secondaries, strict=True):
        assert left.object_id == right.object_id
        assert np.array_equal(left.state.position_m, right.state.position_m)
        assert np.array_equal(left.covariance_ric.matrix, right.covariance_ric.matrix)
    assert first.planted == second.planted


def test_catalog_objects_are_in_plausible_low_earth_orbits() -> None:
    """Every generated object is on a closed orbit above the atmosphere."""
    from conjunction_screening.model.constants import EARTH_RADIUS_M

    catalog = generate_catalog(count=60, planted=6, seed=2024)
    for item in (catalog.primary, *catalog.secondaries):
        elements = item.elements
        assert 0.0 <= elements.eccentricity < 1.0
        assert elements.perigee_radius_m - EARTH_RADIUS_M > 200e3
        assert elements.apogee_radius_m - EARTH_RADIUS_M < 3_100e3
        assert item.radius_m > 0.0
        assert is_symmetric_positive_semidefinite(item.covariance_ric)


def test_generate_catalog_rejects_more_planted_than_objects() -> None:
    """Asking for more planted conjunctions than objects is a configuration error."""
    with pytest.raises(ValueError, match="planted count"):
        generate_catalog(count=3, planted=5)


def test_covariance_stays_semidefinite_through_the_whole_chain(
    conjunction_catalog: SyntheticCatalog,
) -> None:
    """Every representation of the covariance in the pipeline is a valid covariance."""
    from conjunction_screening.algorithm.close_approach import find_close_approaches
    from conjunction_screening.algorithm.filters import CascadeSettings, run_cascade

    settings = CascadeSettings(threshold_m=5_000.0, window_s=conjunction_catalog.window_s)
    for secondary in conjunction_catalog.secondaries[:6]:
        cascade = run_cascade(conjunction_catalog.primary.state, secondary.state, settings)
        approaches = find_close_approaches(
            conjunction_catalog.primary.state,
            secondary.state,
            cascade.candidate_windows,
            CloseApproachSettings(threshold_m=5_000.0),
        )
        assert approaches
        for approach in approaches:
            primary_block, _ = propagated_position_covariance(
                conjunction_catalog.primary, approach.tca_s
            )
            secondary_block, _ = propagated_position_covariance(secondary, approach.tca_s)
            assert is_symmetric_positive_semidefinite(primary_block)
            assert is_symmetric_positive_semidefinite(secondary_block)

            encounter = build_encounter(conjunction_catalog.primary, secondary, approach)
            assert is_symmetric_positive_semidefinite(encounter.relative_covariance)
            assert is_symmetric_positive_semidefinite(encounter.relative_covariance_ric)
            assert is_symmetric_positive_semidefinite(encounter.plane_covariance)


def test_encounter_frames_preserve_the_covariance_eigenvalues(
    conjunction_catalog: SyntheticCatalog,
) -> None:
    """Rotating between the inertial and RIC frames is a rotation, so eigenvalues hold.

    Tolerance: an eigendecomposition of a well-conditioned three by three matrix
    is accurate to a few times machine epsilon relative to the largest
    eigenvalue, so 1e-9 relative is generous.
    """
    from conjunction_screening.algorithm.close_approach import find_close_approaches
    from conjunction_screening.algorithm.filters import CascadeSettings, run_cascade

    secondary = conjunction_catalog.secondaries[0]
    cascade = run_cascade(
        conjunction_catalog.primary.state,
        secondary.state,
        CascadeSettings(threshold_m=5_000.0, window_s=conjunction_catalog.window_s),
    )
    approach = find_close_approaches(
        conjunction_catalog.primary.state,
        secondary.state,
        cascade.candidate_windows,
        CloseApproachSettings(threshold_m=5_000.0),
    )[0]
    encounter = build_encounter(conjunction_catalog.primary, secondary, approach)
    inertial = np.linalg.eigvalsh(encounter.relative_covariance.matrix)
    local = np.linalg.eigvalsh(encounter.relative_covariance_ric.matrix)
    assert np.allclose(inertial, local, rtol=1e-9)


def test_encounter_miss_vector_matches_the_three_dimensional_miss(
    conjunction_catalog: SyntheticCatalog,
) -> None:
    """At the time of closest approach the relative position lies in the encounter plane.

    Tolerance: the relative position makes an angle with the plane of
    ``range rate / relative speed``, and the projection loses a relative fraction
    of half the square of that angle. The bound below is written that way, so it
    tightens automatically if the root find tolerance is tightened.
    """
    from conjunction_screening.algorithm.close_approach import find_close_approaches
    from conjunction_screening.algorithm.filters import CascadeSettings, run_cascade

    for secondary in conjunction_catalog.secondaries[:6]:
        cascade = run_cascade(
            conjunction_catalog.primary.state,
            secondary.state,
            CascadeSettings(threshold_m=5_000.0, window_s=conjunction_catalog.window_s),
        )
        for approach in find_close_approaches(
            conjunction_catalog.primary.state,
            secondary.state,
            cascade.candidate_windows,
            CloseApproachSettings(threshold_m=5_000.0),
        ):
            encounter = build_encounter(conjunction_catalog.primary, secondary, approach)
            angle = abs(approach.range_rate_m_s) / approach.relative_speed_m_s
            allowance = max(0.5 * angle**2, 1e-14)
            assert encounter.projected_miss_distance_m == pytest.approx(
                encounter.miss_distance_m, rel=allowance
            )


def test_screening_is_symmetric_in_the_two_objects(
    conjunction_catalog: SyntheticCatalog,
) -> None:
    """Screening A against B gives the same probability as screening B against A.

    Tolerance: the two runs solve the same root and integrate the same density,
    so they differ only through the order of the floating point operations in the
    frame construction. A relative tolerance of 1e-9 is the quadrature agreement
    tolerance used elsewhere in the suite.
    """
    from conjunction_screening.algorithm.close_approach import find_close_approaches
    from conjunction_screening.algorithm.filters import CascadeSettings, run_cascade

    method = FosterMethod()
    primary = conjunction_catalog.primary
    for secondary in conjunction_catalog.secondaries[:5]:
        cascade = run_cascade(
            primary.state,
            secondary.state,
            CascadeSettings(threshold_m=5_000.0, window_s=conjunction_catalog.window_s),
        )
        approach = find_close_approaches(
            primary.state,
            secondary.state,
            cascade.candidate_windows,
            CloseApproachSettings(threshold_m=5_000.0),
        )[0]
        swapped = find_close_approaches(
            secondary.state,
            primary.state,
            cascade.candidate_windows,
            CloseApproachSettings(threshold_m=5_000.0),
        )[0]
        forward = method.probability(build_encounter(primary, secondary, approach)).value
        backward = method.probability(build_encounter(secondary, primary, swapped)).value
        assert backward == pytest.approx(forward, rel=1e-9)


def test_screening_report_accounts_for_every_object(mixed_catalog: SyntheticCatalog) -> None:
    """Every secondary appears exactly once in the trace, rejected or surviving."""
    report = run_screening(mixed_catalog, ScreeningConfig.for_threshold(5_000.0))
    assert report.screened == mixed_catalog.size
    identifiers = [trace.object_id for trace in report.traces]
    assert identifiers == [item.object_id for item in mixed_catalog.secondaries]
    assert report.survivors + sum(report.rejection_counts.values()) == report.screened
    for trace in report.traces:
        if trace.rejected_by is None:
            assert len(trace.verdicts) == 3
        else:
            assert not trace.verdicts[-1].passed


def test_screening_finds_every_planted_conjunction(
    conjunction_catalog: SyntheticCatalog,
) -> None:
    """A catalogue of nothing but planted conjunctions produces an event for each."""
    report = run_screening(conjunction_catalog, ScreeningConfig.for_threshold(5_000.0))
    found = {event.object_id for event in report.events}
    expected = {truth.object_id for truth in conjunction_catalog.planted}
    assert expected <= found


def test_events_are_sorted_by_decreasing_probability(mixed_catalog: SyntheticCatalog) -> None:
    """The report orders its events the way an operator would read them."""
    report = run_screening(mixed_catalog, ScreeningConfig.for_threshold(5_000.0))
    values = [event.probability.value for event in report.events]
    assert values == sorted(values, reverse=True)


def test_action_thresholds_partition_the_probability_range() -> None:
    """Each probability maps to exactly one action, with the boundaries inclusive below."""
    thresholds = ActionThresholds(act=1e-4, monitor=1e-7)
    assert thresholds.classify(1e-3) is ActionClass.ACT
    assert thresholds.classify(1e-4) is ActionClass.ACT
    assert thresholds.classify(9.9e-5) is ActionClass.MONITOR
    assert thresholds.classify(1e-7) is ActionClass.MONITOR
    assert thresholds.classify(9.9e-8) is ActionClass.DISMISS
    assert thresholds.classify(0.0) is ActionClass.DISMISS


def test_action_thresholds_reject_an_inverted_pair() -> None:
    """A monitor threshold above the act threshold is a configuration error."""
    with pytest.raises(ValueError, match="thresholds"):
        ActionThresholds(act=1e-7, monitor=1e-4)


def test_ranking_is_rendered_as_a_table(regression_report: ScreeningReport) -> None:
    """The ranking table names every column the README quotes."""
    ranked = rank_report(regression_report)
    text = format_ranking_table(ranked, limit=3)
    assert "rank" in text
    assert "Pc" in text
    assert "action" in text
    assert "further event(s) not shown" in text
    assert rank_events(regression_report.events) == ranked


def test_method_comparison_tabulates_every_method() -> None:
    """The comparison table contains one column per method and a disagreement column."""
    from conjunction_screening.model.encounter import planar_encounter

    encounters = {
        "circular": planar_encounter(200.0, 300.0, 300.0, 10.0),
        "elongated": planar_encounter(200.0, 1_500.0, 120.0, 10.0),
    }
    comparisons = compare_methods(
        encounters, (FosterMethod(), AlfanoMethod(), ChanMethod())
    )
    order = (FOSTER, ALFANO, CHAN)
    text = format_comparison_table(comparisons, order)
    for name in order:
        assert name in text
    assert "max rel diff" in text
    assert worst_pairwise_disagreement(comparisons, FOSTER, ALFANO) < 1e-9
    assert comparisons[1].aspect_ratio == pytest.approx(1_500.0 / 120.0, rel=1e-12)


def test_relative_disagreement_handles_two_zeros() -> None:
    """Two identical zeros agree; they do not produce a division by zero."""
    assert relative_disagreement(0.0, 0.0) == 0.0
    assert relative_disagreement(1.0, 0.5) == pytest.approx(0.5)


def test_default_primary_is_a_sun_synchronous_style_orbit() -> None:
    """The reference primary is the orbit the README describes."""
    from conjunction_screening.model.constants import EARTH_RADIUS_M

    primary = default_primary()
    elements = primary.elements
    assert elements.semi_major_axis_m == pytest.approx(EARTH_RADIUS_M + 700e3, rel=1e-9)
    assert np.rad2deg(elements.inclination_rad) == pytest.approx(98.2, abs=1e-6)
    assert primary.radius_m == 5.0
