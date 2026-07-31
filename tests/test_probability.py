"""Property tests for the probability of collision methods.

Tolerances come from the tolerances the methods themselves converge to.

Foster's adaptive quadrature is asked for a relative accuracy of 1e-11 and
Alfano's Simpson refinement for the same, so their difference is bounded by the
sum of those, 2e-11. The tests allow 1e-9, which leaves a factor of fifty for
differences in the platform's exponential and error function implementations,
each of which can differ by an ulp between operating systems.

Chan's series is exact only when the in-plane covariance is circular. Where it is
circular the tolerance is again the quadrature tolerance; where it is not, the
disagreement is the equal-area approximation error and is checked to be present
and bounded rather than absent.

The Monte Carlo comparison uses a band of four binomial standard errors computed
from the estimate itself, never from an observed difference. A four sigma band
fails by chance with probability 6e-5 per case.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from conjunction_screening.algorithm.probability import (
    ALFANO,
    CHAN,
    FOSTER,
    MONTE_CARLO,
    AlfanoMethod,
    ChanMethod,
    FosterMethod,
    MonteCarloMethod,
    ProbabilityMethod,
)
from conjunction_screening.model.encounter import EncounterGeometry, planar_encounter

_QUADRATURE_AGREEMENT = 1e-9
"""Allowed relative disagreement between two quadratures each converged to 1e-11."""

_CASES: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("circular-near", 50.0, 100.0, 100.0, 10.0, 0.0),
    ("circular-mid", 200.0, 250.0, 250.0, 12.0, 0.0),
    ("circular-far", 700.0, 300.0, 300.0, 8.0, 0.0),
    ("elongated-2to1", 150.0, 400.0, 200.0, 10.0, 0.0),
    ("elongated-5to1", 300.0, 1_000.0, 200.0, 12.0, 0.5),
    ("elongated-20to1", 200.0, 2_000.0, 100.0, 15.0, 1.05),
    ("wide-covariance", 120.0, 3_000.0, 1_500.0, 10.0, 0.26),
    ("tight-covariance", 80.0, 60.0, 40.0, 9.0, 0.79),
)


def _encounter(case: tuple[str, float, float, float, float, float]) -> EncounterGeometry:
    _, miss, sigma_x, sigma_y, radius, orientation = case
    return planar_encounter(
        miss_distance_m=miss,
        sigma_x_m=sigma_x,
        sigma_y_m=sigma_y,
        hard_body_radius_m=radius,
        orientation_rad=orientation,
    )


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_foster_and_alfano_agree(case: tuple[str, float, float, float, float, float]) -> None:
    """Two independent formulations of the same integral give the same answer.

    Foster integrates in polar coordinates over the disc with adaptive
    quadrature. Alfano performs the inner integral analytically and applies
    Simpson's rule to the remainder in a substituted variable. Nothing but the
    principal axis reduction is shared, so agreement between them checks both.
    """
    encounter = _encounter(case)
    foster = FosterMethod().probability(encounter)
    alfano = AlfanoMethod().probability(encounter)
    assert foster.converged
    assert alfano.converged
    assert alfano.value == pytest.approx(foster.value, rel=_QUADRATURE_AGREEMENT)


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_every_method_returns_a_probability(
    case: tuple[str, float, float, float, float, float],
) -> None:
    """All results lie in the unit interval and report their own accuracy."""
    encounter = _encounter(case)
    methods: tuple[ProbabilityMethod, ...] = (FosterMethod(), AlfanoMethod(), ChanMethod())
    for method in methods:
        result = method.probability(encounter)
        assert 0.0 <= result.value <= 1.0
        assert result.method == method.name
        assert result.error_estimate >= 0.0
        assert result.detail


@pytest.mark.parametrize("miss_distance_m", [20.0, 100.0, 400.0, 900.0])
@pytest.mark.parametrize("sigma_m", [80.0, 300.0])
def test_chan_is_exact_for_a_circular_covariance(miss_distance_m: float, sigma_m: float) -> None:
    """Chan's equal-area substitution is an identity when the covariance is circular.

    In that case the scaled hard body circle is already a circle, so the series
    is an exact evaluation of the non-central chi-square tail rather than an
    approximation, and it must agree with the quadrature to the quadrature's own
    tolerance.
    """
    encounter = planar_encounter(
        miss_distance_m=miss_distance_m,
        sigma_x_m=sigma_m,
        sigma_y_m=sigma_m,
        hard_body_radius_m=10.0,
    )
    foster = FosterMethod().probability(encounter)
    chan = ChanMethod().probability(encounter)
    assert chan.converged
    assert chan.value == pytest.approx(foster.value, rel=1e-6)


def test_chan_disagreement_grows_with_the_aspect_ratio() -> None:
    """The equal-area approximation degrades as the covariance becomes more elongated.

    This is a property of Chan's derivation, not a defect. Recording it is what
    justifies using Chan as a cross check rather than as the primary result.
    """
    disagreements = []
    for aspect in (1.0, 4.0, 20.0):
        encounter = planar_encounter(
            miss_distance_m=250.0,
            sigma_x_m=400.0 * float(np.sqrt(aspect)),
            sigma_y_m=400.0 / float(np.sqrt(aspect)),
            hard_body_radius_m=12.0,
        )
        foster = FosterMethod().probability(encounter).value
        chan = ChanMethod().probability(encounter).value
        disagreements.append(abs(chan - foster) / foster)
    assert disagreements[0] < 1e-9
    assert disagreements[0] < disagreements[1] < disagreements[2]
    assert disagreements[2] < 0.2


def test_probability_falls_to_zero_as_the_miss_distance_grows() -> None:
    """Pushing the encounter further away drives the probability monotonically down."""
    method = FosterMethod()
    values = [
        method.probability(
            planar_encounter(
                miss_distance_m=distance,
                sigma_x_m=250.0,
                sigma_y_m=250.0,
                hard_body_radius_m=10.0,
            )
        ).value
        for distance in (10.0, 100.0, 400.0, 1_000.0, 2_500.0, 6_000.0)
    ]
    assert all(later < earlier for earlier, later in pairwise(values))
    assert values[-1] < 1e-30


def test_probability_rises_towards_one_as_the_hard_body_radius_grows() -> None:
    """A disc that eventually covers the whole density collects all of it."""
    method = FosterMethod()
    base = planar_encounter(
        miss_distance_m=100.0, sigma_x_m=200.0, sigma_y_m=120.0, hard_body_radius_m=1.0
    )
    radii = (1.0, 10.0, 50.0, 200.0, 1_000.0, 4_000.0)
    values = [method.probability(base.with_hard_body_radius(radius)).value for radius in radii]
    assert all(later > earlier for earlier, later in pairwise(values))
    assert values[-1] > 0.999


def test_probability_grows_quadratically_with_a_small_hard_body_radius() -> None:
    """For a radius far below the covariance scale the disc integral is area times density.

    Doubling the radius must therefore quadruple the probability. Tolerance: the
    next term of the expansion is smaller by ``(R / sigma)^2``, which is 4e-4 for
    the largest radius used here, so a relative tolerance of 1e-2 is well clear.
    """
    method = FosterMethod()
    base = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=400.0, sigma_y_m=300.0, hard_body_radius_m=1.0
    )
    small = method.probability(base.with_hard_body_radius(4.0)).value
    large = method.probability(base.with_hard_body_radius(8.0)).value
    assert large / small == pytest.approx(4.0, rel=1e-2)


@pytest.mark.parametrize(
    "case",
    [case for case in _CASES if case[0] in {"circular-near", "elongated-2to1", "tight-covariance"}],
    ids=["circular-near", "elongated-2to1", "tight-covariance"],
)
def test_monte_carlo_agrees_with_the_analytic_value(
    case: tuple[str, float, float, float, float, float],
) -> None:
    """Sampling the three-dimensional encounter reproduces the analytic probability.

    This is the strongest check available, because the Monte Carlo estimate goes
    through the covariance square root, the projection onto the plane normal to
    the relative velocity, and the hard body test, none of which the quadrature
    touches.

    Tolerance: four binomial standard errors, computed from the estimate itself.
    That band is exceeded by chance with probability 6e-5 for a correct
    implementation, and it shrinks as the sample count grows, so it cannot be
    satisfied by an implementation that is merely close.
    """
    encounter = _encounter(case)
    analytic = FosterMethod().probability(encounter)
    sampled = MonteCarloMethod(samples=1_000_000, seed=515).probability(encounter)
    assert sampled.converged
    assert abs(sampled.value - analytic.value) <= 4.0 * sampled.error_estimate


def test_monte_carlo_standard_error_shrinks_with_the_sample_count() -> None:
    """The reported standard error follows the one over root n law."""
    encounter = _encounter(_CASES[0])
    small = MonteCarloMethod(samples=100_000, seed=808).probability(encounter)
    large = MonteCarloMethod(samples=400_000, seed=808).probability(encounter)
    assert large.error_estimate == pytest.approx(0.5 * small.error_estimate, rel=0.1)


def test_monte_carlo_reports_a_starved_estimate_as_not_converged() -> None:
    """Too few hits gives an estimate with no useful precision, and it says so."""
    encounter = planar_encounter(
        miss_distance_m=3_000.0, sigma_x_m=200.0, sigma_y_m=200.0, hard_body_radius_m=5.0
    )
    result = MonteCarloMethod(samples=20_000, seed=17).probability(encounter)
    assert not result.converged


def test_methods_expose_their_names() -> None:
    """Every method reports the identifier used to key comparison tables."""
    assert FosterMethod().name == FOSTER
    assert AlfanoMethod().name == ALFANO
    assert ChanMethod().name == CHAN
    assert MonteCarloMethod().name == MONTE_CARLO


def test_singular_in_plane_covariance_is_rejected() -> None:
    """A degenerate covariance gives an improper density and no probability."""
    encounter = planar_encounter(
        miss_distance_m=100.0, sigma_x_m=200.0, sigma_y_m=200.0, hard_body_radius_m=5.0
    )
    degenerate = encounter.with_scaled_covariance(1.0)
    from conjunction_screening.model.covariance import Covariance

    broken = EncounterGeometry(
        tca_s=degenerate.tca_s,
        relative_position_m=degenerate.relative_position_m,
        relative_velocity_m_s=degenerate.relative_velocity_m_s,
        relative_covariance=degenerate.relative_covariance,
        relative_covariance_ric=degenerate.relative_covariance_ric,
        basis=degenerate.basis,
        miss_vector_m=degenerate.miss_vector_m,
        plane_covariance=Covariance(matrix=np.zeros((2, 2)), frame="encounter"),
        hard_body_radius_m=degenerate.hard_body_radius_m,
    )
    with pytest.raises(ValueError, match="singular"):
        FosterMethod().probability(broken)
