"""Maximum probability of collision over a scaling of the covariance.

The probability of collision is not monotonic in the size of the covariance. For
a fixed miss distance it rises as the uncertainty grows, because the hard body
disc moves into the bulk of the density, and then falls, because the density
spreads out faster than the disc gains from being nearer the centre. The peak of
that curve is the largest probability consistent with the observed miss distance
while the shape of the covariance is held fixed, and it is the quantity that
tells an operator whether a conjunction can be dismissed on covariance grounds
alone.

For a circular in-plane covariance of standard deviation ``s``, and a hard body
radius small enough that the density is effectively constant across the disc,

    Pc = (R^2 / (2 s^2)) exp(-d^2 / (2 s^2))

which is maximised at ``s = d / sqrt(2)`` with value ``R^2 / (e d^2)``. That
closed form, from Alfano (2005b), is the analytic reference the numerical search
is checked against.

The reference applies to circular covariances and is not a bound over all of
them. Shrinking one principal axis concentrates the same probability mass into a
narrower band, so an elongated covariance whose wide axis lies along the miss
vector can reach a higher probability than the circular value. That is why the
search scales a covariance rather than replacing it: the answer is only
meaningful relative to a fixed covariance shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import minimize_scalar

from conjunction_screening.algorithm.probability import ProbabilityMethod
from conjunction_screening.model.encounter import EncounterGeometry, principal_axis_form

__all__ = [
    "MaximumProbability",
    "isotropic_maximum_probability",
    "isotropic_maximum_sigma_m",
    "maximum_probability",
]

_LOG_TEN: Final[float] = float(np.log(10.0))


@dataclass(frozen=True, slots=True)
class MaximumProbability:
    """The peak of the probability curve over covariance scaling.

    Attributes:
        scale: Multiple of the nominal standard deviations at which the peak
            occurs, dimensionless.
        probability: Probability at the peak.
        sigma_x_m: Larger principal in-plane standard deviation at the peak, in m.
        sigma_y_m: Smaller principal in-plane standard deviation at the peak, in m.
        method: Identifier of the probability method used.
        converged: Whether the optimiser reported success and the probability
            evaluation at the peak converged.
        evaluations: Number of probability evaluations the search used.
    """

    scale: float
    probability: float
    sigma_x_m: float
    sigma_y_m: float
    method: str
    converged: bool
    evaluations: int


def isotropic_maximum_probability(miss_distance_m: float, hard_body_radius_m: float) -> float:
    """Return ``R^2 / (e d^2)``, the small-radius maximum for a circular covariance.

    Valid for circular in-plane covariances. An elongated covariance can exceed
    it, so this is a reference value and not an upper bound.
    """
    if not miss_distance_m > 0.0:
        raise ValueError("miss distance must be positive")
    return float(hard_body_radius_m**2 / (np.e * miss_distance_m**2))


def isotropic_maximum_sigma_m(miss_distance_m: float) -> float:
    """Return ``d / sqrt(2)``, the standard deviation that maximises the probability."""
    return float(miss_distance_m / np.sqrt(2.0))


def maximum_probability(
    encounter: EncounterGeometry,
    method: ProbabilityMethod,
    lower_scale: float = 1e-3,
    upper_scale: float = 1e3,
    tolerance: float = 1e-9,
) -> MaximumProbability:
    """Maximise the probability of collision over an isotropic scaling of the covariance.

    The search runs in the base ten logarithm of the scale, because the curve is
    close to symmetric in that variable and spans several decades. Brent's
    bounded method is used, which needs no derivatives and converges on a
    unimodal objective; the curve is unimodal in the scale for a fixed geometry.

    Args:
        encounter: Encounter whose covariance is scaled. The miss vector, the
            relative velocity, and the hard body radius are held fixed.
        method: Probability method evaluated at each trial scale.
        lower_scale: Smallest multiple of the nominal covariance considered.
        upper_scale: Largest multiple considered.
        tolerance: Absolute tolerance on the logarithm of the scale. The
            resulting tolerance on the peak probability is second order in this
            value, because the derivative vanishes at a peak.

    Returns:
        The location and value of the peak.
    """
    if not 0.0 < lower_scale < upper_scale:
        raise ValueError("scale bounds must satisfy 0 < lower_scale < upper_scale")

    evaluations = 0
    converged_everywhere = True

    def negative(log_scale: float) -> float:
        nonlocal evaluations, converged_everywhere
        evaluations += 1
        result = method.probability(encounter.with_scaled_covariance(float(10.0**log_scale)))
        converged_everywhere = converged_everywhere and result.converged
        return -result.value

    outcome = minimize_scalar(
        negative,
        bounds=(float(np.log10(lower_scale)), float(np.log10(upper_scale))),
        method="bounded",
        options={"xatol": tolerance / _LOG_TEN},
    )
    scale = float(10.0 ** float(outcome.x))
    peak = encounter.with_scaled_covariance(scale)
    form = principal_axis_form(peak)
    final = method.probability(peak)
    evaluations += 1
    return MaximumProbability(
        scale=scale,
        probability=final.value,
        sigma_x_m=form.sigma_x_m,
        sigma_y_m=form.sigma_y_m,
        method=method.name,
        converged=bool(outcome.success) and final.converged and converged_everywhere,
        evaluations=evaluations,
    )
