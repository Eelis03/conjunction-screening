"""Regression tests pinning one recorded screening run.

What is pinned and why it is safe to pin:

* Counts, filter verdicts, and the ranking order are discrete. They are pinned
  exactly, and a separate test checks that no filter decision in this run sits
  near its threshold, so a one ulp difference on another machine cannot flip one.
* Times of closest approach and miss distances come from a root find that
  reported convergence. A non-converged iterate is excluded from the report by
  ``run_screening``, and this module asserts that every pinned event converged,
  because the state of a non-converged iteration depends on floating point
  reduction order and is not reproducible across machines.
* Probabilities come from an adaptive quadrature that met a relative tolerance of
  1e-11.

Where the tolerances come from:

* Time of closest approach: the root find tolerance is 1e-9 s and the propagator
  round trip contributes about the same, so the value is determined to roughly
  1e-9 s. The pinned tolerance of 1e-4 s is five orders above that, which absorbs
  differences between the platforms' trigonometric implementations while still
  fixing nine significant figures of a value of order 1e4 s.
* Miss distance: the dominant error is the propagator round trip, of order 1e-7 m
  on distances between 1e2 and 4e3 m, so about 1e-9 relative. The pinned
  tolerance of 1e-7 relative leaves two orders.
* Probability: the quadrature converges to 1e-11 relative. A one ulp difference
  in the platform exponential is amplified by the magnitude of the exponent,
  which for the smallest pinned value here is about 515, giving 5e-13. The pinned
  tolerance of 1e-6 relative leaves six orders. The absolute floor of 1e-250
  applies to values that have underflowed; they carry no decision information and
  are compared absolutely rather than relatively.
"""

from __future__ import annotations

import pytest

from conjunction_screening.analysis.ranking import ActionThresholds, rank_report
from conjunction_screening.pipeline.screening import ScreeningReport

_EXPECTED_SCREENED = 120
_EXPECTED_REJECTIONS = {"perigee-apogee": 95, "orbit-path": 16, "time": 1}
_EXPECTED_SURVIVORS = 8
_EXPECTED_CANDIDATE_SECONDS = 2792.133927

# object id, time of closest approach in s, miss distance in m, relative speed in
# m/s, combined hard body radius in m, probability of collision
_EXPECTED_EVENTS: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("PLANTED-05", 12353.126016474389, 122.47183526904838,
     744.1590450784526, 6.730542619579624, 1.4379280656158053e-04),
    ("PLANTED-06", 67252.0059954321, 103.51667155344157,
     1670.3907113558985, 7.832254968939143, 1.3852067078963632e-04),
    ("PLANTED-01", 57650.35869239602, 230.71284102973127,
     2644.6710966733494, 7.315816285144148, 1.0201133586561593e-05),
    ("PLANTED-02", 21248.42732437559, 1997.8467165951122,
     471.0043914744732, 6.826285933000365, 1.3181321797307335e-10),
    ("PLANTED-08", 73496.49461152835, 907.9239650841437,
     786.3347064845029, 5.758000930212429, 8.373401070614024e-19),
    ("PLANTED-07", 32526.929379558343, 500.8975021108618,
     4575.973418748839, 5.612141596499105, 2.1837769938871676e-19),
    ("PLANTED-04", 66519.20206735266, 1710.6533065971628,
     2938.404648888308, 7.386580231430883, 4.58124931705702e-224),
    ("PLANTED-03", 53295.10240118718, 3382.896316238055,
     572.2715050612472, 6.432127918434734, 0.0),
)

_EXPECTED_ACTIONS: tuple[tuple[str, str], ...] = (
    ("PLANTED-05", "act"),
    ("PLANTED-06", "act"),
    ("PLANTED-01", "monitor"),
    ("PLANTED-02", "dismiss"),
    ("PLANTED-08", "dismiss"),
    ("PLANTED-07", "dismiss"),
    ("PLANTED-04", "dismiss"),
    ("PLANTED-03", "dismiss"),
)

_TIME_TOLERANCE_S = 1e-4
_DISTANCE_RELATIVE_TOLERANCE = 1e-7
_PROBABILITY_RELATIVE_TOLERANCE = 1e-6
_PROBABILITY_ABSOLUTE_FLOOR = 1e-250


def test_filter_counts_are_unchanged(regression_report: ScreeningReport) -> None:
    """The cascade rejects the same pairs at the same stages."""
    assert regression_report.screened == _EXPECTED_SCREENED
    assert regression_report.rejection_counts == _EXPECTED_REJECTIONS
    assert regression_report.survivors == _EXPECTED_SURVIVORS
    assert sum(_EXPECTED_REJECTIONS.values()) + _EXPECTED_SURVIVORS == _EXPECTED_SCREENED


def test_candidate_window_total_is_unchanged(regression_report: ScreeningReport) -> None:
    """The time filter narrows the search to the same total duration.

    Tolerance: window edges are computed from Kepler's equation on values of
    order 1e4 s, so their relative error is a few times machine epsilon. A
    relative tolerance of 1e-9 pins nine significant figures.
    """
    assert regression_report.cascade_cost_windows == pytest.approx(
        _EXPECTED_CANDIDATE_SECONDS, rel=1e-9
    )


def test_every_pinned_event_converged(regression_report: ScreeningReport) -> None:
    """Nothing in the baseline rests on a non-converged solve.

    This is the precondition for the rest of this module. A close approach whose
    refinement stopped on its iteration cap, or a quadrature that missed its
    tolerance, would carry a value that depends on the order a floating point
    reduction happened to run in, and pinning it would produce a test that passes
    on the machine that recorded it and fails elsewhere.
    """
    assert regression_report.events
    for event in regression_report.events:
        assert event.approach.converged, event.object_id
        assert event.probability.converged, event.object_id


def test_events_match_the_recorded_run(regression_report: ScreeningReport) -> None:
    """Every reported conjunction reproduces its recorded values."""
    assert len(regression_report.events) == len(_EXPECTED_EVENTS)
    for event, expected in zip(regression_report.events, _EXPECTED_EVENTS, strict=True):
        object_id, tca_s, miss_m, speed, radius, probability = expected
        assert event.object_id == object_id
        assert event.tca_s == pytest.approx(tca_s, abs=_TIME_TOLERANCE_S)
        assert event.miss_distance_m == pytest.approx(
            miss_m, rel=_DISTANCE_RELATIVE_TOLERANCE
        )
        assert event.relative_speed_m_s == pytest.approx(speed, rel=1e-9)
        assert event.encounter.hard_body_radius_m == pytest.approx(radius, rel=1e-12)
        assert event.probability.value == pytest.approx(
            probability,
            rel=_PROBABILITY_RELATIVE_TOLERANCE,
            abs=_PROBABILITY_ABSOLUTE_FLOOR,
        )


def test_ranking_and_actions_are_unchanged(regression_report: ScreeningReport) -> None:
    """The ranked order and the action each event is escalated to are unchanged."""
    ranked = rank_report(regression_report, ActionThresholds())
    assert tuple((item.object_id, item.action.value) for item in ranked) == _EXPECTED_ACTIONS
    assert tuple(item.rank for item in ranked) == tuple(range(1, len(ranked) + 1))
    probabilities = [item.probability for item in ranked]
    assert probabilities == sorted(probabilities, reverse=True)


def test_no_filter_decision_sits_near_its_threshold(regression_report: ScreeningReport) -> None:
    """No pinned count depends on a comparison that a rounding difference could flip.

    Every filter verdict is an inequality between a computed bound and a
    threshold. If one of those comparisons were marginal, the counts pinned above
    would be reproducing an accident rather than a result. This test measures the
    tightest margin in the recorded run and requires it to be a substantial
    fraction of the scale of the comparison, so the pinned counts stand on
    geometry rather than on the last bits of a float.
    """
    tightest_distance = min(
        abs(verdict.bound - verdict.threshold)
        for trace in regression_report.traces
        for verdict in trace.verdicts
        if verdict.units == "m" and not verdict.passed
    )
    assert tightest_distance > 0.1 * regression_report.threshold_m

    tightest_time = min(
        abs(verdict.bound - verdict.threshold)
        for trace in regression_report.traces
        for verdict in trace.verdicts
        if verdict.units == "s" and not verdict.passed
    )
    assert tightest_time > 60.0


def test_screening_run_is_deterministic(regression_report: ScreeningReport) -> None:
    """Two runs of the same configuration produce identical reports.

    Determinism within a process is a weaker claim than reproducibility across
    machines, but it is the one that catches an accidental dependence on a global
    generator or on iteration order.
    """
    from conjunction_screening.pipeline.catalog import generate_catalog
    from conjunction_screening.pipeline.screening import ScreeningConfig, run_screening

    repeat = run_screening(
        generate_catalog(count=120, planted=8, window_s=86_400.0, seed=20260731),
        ScreeningConfig.for_threshold(5_000.0),
    )
    assert [event.object_id for event in repeat.events] == [
        event.object_id for event in regression_report.events
    ]
    for first, second in zip(regression_report.events, repeat.events, strict=True):
        assert first.tca_s == second.tca_s
        assert first.miss_distance_m == second.miss_distance_m
        assert first.probability.value == second.probability.value
