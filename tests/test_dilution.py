"""Property tests for the dilution behaviour and the maximum probability.

The closed form used as a reference here is the small-radius limit for a circular
covariance,

    Pc = (R^2 / (2 s^2)) exp(-d^2 / (2 s^2))

maximised at ``s = d / sqrt(2)`` with value ``R^2 / (e d^2)``. Its relative error
is of order ``(R / s)^2`` at the peak, which is ``2 (R / d)^2``. Every tolerance
against that reference is written in those terms, so a test that uses a different
radius to distance ratio carries a different and correct tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjunction_screening.algorithm.maximum import (
    isotropic_maximum_probability,
    isotropic_maximum_sigma_m,
    maximum_probability,
)
from conjunction_screening.algorithm.probability import FosterMethod
from conjunction_screening.analysis.dilution import dilution_curve, format_dilution_summary
from conjunction_screening.model.encounter import planar_encounter


def _closed_form_relative_error(radius_m: float, miss_distance_m: float) -> float:
    """Return the relative error of the point-mass closed form at its own peak."""
    return 2.0 * (radius_m / miss_distance_m) ** 2


@pytest.mark.parametrize(
    ("miss_distance_m", "radius_m"),
    [(1_000.0, 10.0), (500.0, 5.0), (2_000.0, 10.0), (300.0, 3.0)],
)
def test_maximum_probability_matches_the_closed_form(
    miss_distance_m: float, radius_m: float
) -> None:
    """The numerical maximum over covariance scaling reproduces ``R^2 / (e d^2)``.

    Tolerance: the closed form is the limit of a vanishing hard body radius, and
    its leading correction is ``2 (R / d)^2``. A factor of ten is allowed on top
    of that, which for the tightest case here is 2e-3.
    """
    nominal_sigma = miss_distance_m
    encounter = planar_encounter(
        miss_distance_m=miss_distance_m,
        sigma_x_m=nominal_sigma,
        sigma_y_m=nominal_sigma,
        hard_body_radius_m=radius_m,
    )
    peak = maximum_probability(encounter, FosterMethod(), lower_scale=1e-3, upper_scale=1e3)
    assert peak.converged

    expected_probability = isotropic_maximum_probability(miss_distance_m, radius_m)
    expected_sigma = isotropic_maximum_sigma_m(miss_distance_m)
    allowance = 10.0 * _closed_form_relative_error(radius_m, miss_distance_m)

    assert peak.probability == pytest.approx(expected_probability, rel=allowance)
    assert peak.sigma_x_m == pytest.approx(expected_sigma, rel=allowance)
    assert peak.sigma_y_m == pytest.approx(expected_sigma, rel=allowance)
    assert peak.scale == pytest.approx(expected_sigma / nominal_sigma, rel=allowance)


def test_maximum_probability_dominates_every_sampled_scale() -> None:
    """The optimiser really finds a maximum of the curve it is sampled against."""
    encounter = planar_encounter(
        miss_distance_m=400.0, sigma_x_m=900.0, sigma_y_m=180.0, hard_body_radius_m=12.0
    )
    method = FosterMethod()
    curve = dilution_curve(encounter, method, minimum_scale=1e-2, maximum_scale=1e3, points=61)
    assert curve.peak.probability >= float(np.max(curve.probabilities)) * (1.0 - 1e-9)


def test_probability_rises_then_falls_as_the_covariance_is_inflated() -> None:
    """The dilution behaviour itself: not monotonic in covariance size.

    The nominal covariance is set well below the value that maximises the
    probability, so the sampled curve must contain an interior maximum with a
    strictly rising branch before it and a strictly falling branch after it.
    """
    encounter = planar_encounter(
        miss_distance_m=500.0, sigma_x_m=50.0, sigma_y_m=50.0, hard_body_radius_m=10.0
    )
    curve = dilution_curve(
        encounter, FosterMethod(), minimum_scale=1e-1, maximum_scale=1e3, points=81
    )
    usable = curve.probabilities > 0.0
    values = curve.probabilities[usable]
    peak_index = int(np.argmax(values))
    assert 0 < peak_index < len(values) - 1

    rising = values[: peak_index + 1]
    falling = values[peak_index:]
    assert np.all(np.diff(rising) > 0.0)
    assert np.all(np.diff(falling) < 0.0)
    assert values[-1] < values[peak_index] / 100.0


def test_falling_branch_follows_the_inverse_square_asymptote() -> None:
    """Far into the dilution region the probability behaves as ``R^2 / (2 s^2)``.

    Tolerance: the neglected exponential factor contributes a slope correction of
    order ``d^2 / s^2``, which over the largest decade sampled here is below
    1e-4, so a slope within 0.01 of minus two is a strong check.
    """
    encounter = planar_encounter(
        miss_distance_m=300.0, sigma_x_m=300.0, sigma_y_m=300.0, hard_body_radius_m=10.0
    )
    curve = dilution_curve(
        encounter, FosterMethod(), minimum_scale=1e0, maximum_scale=1e4, points=81
    )
    assert curve.falling_branch_slope == pytest.approx(-2.0, abs=1e-2)


def test_dilution_factor_reports_how_far_the_covariance_sits_past_the_peak() -> None:
    """An over-inflated covariance reports a probability well below the achievable maximum."""
    encounter = planar_encounter(
        miss_distance_m=100.0, sigma_x_m=2_000.0, sigma_y_m=2_000.0, hard_body_radius_m=10.0
    )
    curve = dilution_curve(
        encounter, FosterMethod(), minimum_scale=1e-3, maximum_scale=1e2, points=61
    )
    assert curve.in_dilution_region
    assert curve.dilution_factor > 100.0
    assert curve.peak.scale < 1.0


def test_a_covariance_below_the_peak_is_not_in_the_dilution_region() -> None:
    """A tight covariance sits on the rising branch, where more uncertainty raises Pc."""
    encounter = planar_encounter(
        miss_distance_m=1_000.0, sigma_x_m=50.0, sigma_y_m=50.0, hard_body_radius_m=10.0
    )
    curve = dilution_curve(
        encounter, FosterMethod(), minimum_scale=1e-1, maximum_scale=1e3, points=61
    )
    assert not curve.in_dilution_region
    assert curve.peak.scale > 1.0


def test_an_elongated_covariance_can_exceed_the_circular_reference() -> None:
    """``R^2 / (e d^2)`` bounds circular covariances only, and elongation beats it.

    The closed form is the maximum over the family of circular covariances. It is
    not a bound over all covariances: shrinking one principal axis concentrates
    the same probability mass into a narrower band, and if the miss vector lies
    along the wide axis the hard body disc sits in a region of higher density. The
    reference value is therefore a comparison point rather than a ceiling, and the
    library reports it as such.
    """
    encounter = planar_encounter(
        miss_distance_m=400.0, sigma_x_m=4_000.0, sigma_y_m=100.0, hard_body_radius_m=10.0
    )
    curve = dilution_curve(
        encounter, FosterMethod(), minimum_scale=1e-2, maximum_scale=1e3, points=61
    )
    assert curve.peak.probability > curve.analytic_peak_probability
    assert curve.analytic_peak_probability == pytest.approx(
        isotropic_maximum_probability(400.0, 10.0), rel=1e-12
    )


def test_the_circular_reference_is_attained_only_by_a_circular_covariance() -> None:
    """Among covariances of equal determinant, the circular one is the stationary case.

    Scaling a circular covariance reaches exactly the closed form, which the
    earlier test pins. This one records the complementary fact that an elongated
    covariance of the same determinant reaches a different value, so the two
    together show the closed form is specific to the circular family.
    """
    method = FosterMethod()
    circular = planar_encounter(
        miss_distance_m=600.0, sigma_x_m=600.0, sigma_y_m=600.0, hard_body_radius_m=6.0
    )
    elongated = planar_encounter(
        miss_distance_m=600.0, sigma_x_m=2_400.0, sigma_y_m=150.0, hard_body_radius_m=6.0
    )
    circular_peak = maximum_probability(circular, method).probability
    elongated_peak = maximum_probability(elongated, method).probability
    reference = isotropic_maximum_probability(600.0, 6.0)
    allowance = 10.0 * _closed_form_relative_error(6.0, 600.0)
    assert circular_peak == pytest.approx(reference, rel=allowance)
    assert abs(elongated_peak - reference) > 0.05 * reference


def test_dilution_summary_is_renderable() -> None:
    """The text summary covers every reported quantity."""
    encounter = planar_encounter(
        miss_distance_m=250.0, sigma_x_m=600.0, sigma_y_m=200.0, hard_body_radius_m=9.0
    )
    curve = dilution_curve(
        encounter, FosterMethod(), minimum_scale=1e-1, maximum_scale=1e2, points=21
    )
    text = format_dilution_summary(curve)
    for label in ("nominal Pc", "peak Pc", "analytic peak Pc", "dilution factor"):
        assert label in text


def test_dilution_curve_rejects_invalid_bounds() -> None:
    """Configuration errors are refused rather than producing a meaningless curve."""
    encounter = planar_encounter(
        miss_distance_m=100.0, sigma_x_m=100.0, sigma_y_m=100.0, hard_body_radius_m=5.0
    )
    with pytest.raises(ValueError, match="scale bounds"):
        dilution_curve(encounter, FosterMethod(), minimum_scale=10.0, maximum_scale=1.0)
    with pytest.raises(ValueError, match="points"):
        dilution_curve(encounter, FosterMethod(), points=2)
