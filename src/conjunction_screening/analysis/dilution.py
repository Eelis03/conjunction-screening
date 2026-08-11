"""The probability dilution study.

Probability of collision is not monotonic in the size of the covariance. Holding
the miss distance and the hard body radius fixed and inflating the covariance,
the probability first rises, peaks, and then falls. The falling branch is the
dilution region: a worse orbit determination solution reports a smaller
probability of collision, so a screening system that trusts small probabilities
without also checking the covariance can dismiss a conjunction precisely because
it knows too little about it.

The behaviour is easy to see in the small-radius limit for a circular covariance
of standard deviation ``s``:

    Pc = (R^2 / (2 s^2)) exp(-d^2 / (2 s^2))

The exponential dominates while ``s < d / sqrt(2)`` and the ``1 / s^2`` prefactor
dominates after it. The peak sits at ``s = d / sqrt(2)`` with value
``R^2 / (e d^2)``. That value is the maximum over circular covariances, not over
all of them: an elongated covariance concentrates the same mass into a narrower
band and can exceed it. This module computes the curve numerically for a real
encounter, locates the peak, compares it against the circular reference, and
quantifies how far into the dilution region a given covariance sits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conjunction_screening.algorithm.maximum import (
    MaximumProbability,
    maximum_probability,
)
from conjunction_screening.algorithm.probability import ProbabilityMethod
from conjunction_screening.model.arrays import Vector
from conjunction_screening.model.encounter import EncounterGeometry, principal_axis_form

__all__ = ["DilutionCurve", "dilution_curve", "format_dilution_summary"]


@dataclass(frozen=True, slots=True)
class DilutionCurve:
    """Probability of collision as a function of covariance scale.

    Attributes:
        scales: Multiples of the nominal covariance standard deviations.
        probabilities: Probability at each scale.
        nominal_probability: Probability at a scale of one.
        peak: The numerically located maximum over the scale.
        analytic_peak_probability: The closed-form ``R^2 / (e d^2)`` reference,
            which is the maximum for a circular covariance of the same miss
            distance and hard body radius. An elongated covariance can exceed it.
        analytic_peak_sigma_m: The closed-form ``d / sqrt(2)`` reference.
        nominal_sigma_x_m: Larger principal in-plane sigma at unit scale, in m.
        nominal_sigma_y_m: Smaller principal in-plane sigma at unit scale, in m.
        miss_distance_m: Projected miss distance, in m.
        hard_body_radius_m: Combined hard body radius, in m.
        method: Identifier of the probability method used.
    """

    scales: Vector
    probabilities: Vector
    nominal_probability: float
    peak: MaximumProbability
    analytic_peak_probability: float
    analytic_peak_sigma_m: float
    nominal_sigma_x_m: float
    nominal_sigma_y_m: float
    miss_distance_m: float
    hard_body_radius_m: float
    method: str

    @property
    def dilution_factor(self) -> float:
        """Ratio of the peak probability to the probability at the nominal covariance.

        A value far above one means the nominal covariance sits well away from
        the peak, so the reported probability understates what the same miss
        distance could support under a different uncertainty.
        """
        if self.nominal_probability <= 0.0:
            return float("inf")
        return self.peak.probability / self.nominal_probability

    @property
    def in_dilution_region(self) -> bool:
        """True when the nominal covariance is larger than the one that maximises Pc."""
        return self.peak.scale < 1.0

    @property
    def falling_branch_slope(self) -> float:
        """Fitted slope of log Pc against log scale over the largest decade sampled.

        The small-radius analysis predicts a slope of minus two once the
        exponential factor has saturated, because the probability then behaves as
        ``R^2 / (2 s^2)``. Measuring the slope is a check that the numerical curve
        reaches the predicted asymptote.
        """
        mask = (self.scales >= self.scales[-1] / 10.0) & (self.probabilities > 0.0)
        if int(np.count_nonzero(mask)) < 2:
            return float("nan")
        slope = np.polyfit(np.log(self.scales[mask]), np.log(self.probabilities[mask]), 1)[0]
        return float(slope)


def dilution_curve(
    encounter: EncounterGeometry,
    method: ProbabilityMethod,
    minimum_scale: float = 1e-2,
    maximum_scale: float = 1e3,
    points: int = 121,
) -> DilutionCurve:
    """Evaluate the probability over a logarithmic sweep of the covariance scale.

    Args:
        encounter: Encounter whose covariance is inflated. The miss vector, the
            relative velocity, and the hard body radius are held fixed, which is
            what isolates the covariance as the only varying quantity.
        method: Probability method evaluated at each scale.
        minimum_scale: Smallest multiple of the nominal covariance sampled.
        maximum_scale: Largest multiple sampled.
        points: Number of logarithmically spaced samples.

    Returns:
        The curve, its peak, and the analytic references it is compared against.
    """
    if not 0.0 < minimum_scale < maximum_scale:
        raise ValueError("scale bounds must satisfy 0 < minimum_scale < maximum_scale")
    if points < 3:
        raise ValueError("points must be at least 3")

    scales = np.logspace(np.log10(minimum_scale), np.log10(maximum_scale), points)
    probabilities = np.array(
        [method.probability(encounter.with_scaled_covariance(float(s))).value for s in scales],
        dtype=np.float64,
    )
    nominal = method.probability(encounter).value
    peak = maximum_probability(
        encounter, method, lower_scale=minimum_scale, upper_scale=maximum_scale
    )
    form = principal_axis_form(encounter)
    distance = encounter.projected_miss_distance_m
    radius = encounter.hard_body_radius_m
    return DilutionCurve(
        scales=scales,
        probabilities=probabilities,
        nominal_probability=float(nominal),
        peak=peak,
        analytic_peak_probability=float(radius**2 / (np.e * distance**2)),
        analytic_peak_sigma_m=float(distance / np.sqrt(2.0)),
        nominal_sigma_x_m=form.sigma_x_m,
        nominal_sigma_y_m=form.sigma_y_m,
        miss_distance_m=distance,
        hard_body_radius_m=radius,
        method=method.name,
    )


def format_dilution_summary(curve: DilutionCurve) -> str:
    """Render the key numbers of a dilution study as text."""
    peak_sigma = float(np.sqrt(curve.peak.sigma_x_m * curve.peak.sigma_y_m))
    lines = [
        f"method                     {curve.method}",
        f"miss distance              {curve.miss_distance_m:.1f} m",
        f"hard body radius           {curve.hard_body_radius_m:.1f} m",
        f"nominal in-plane sigmas    {curve.nominal_sigma_x_m:.1f} m by "
        f"{curve.nominal_sigma_y_m:.1f} m",
        f"nominal Pc                 {curve.nominal_probability:.6e}",
        f"peak Pc                    {curve.peak.probability:.6e} at scale {curve.peak.scale:.4f}",
        f"peak geometric mean sigma  {peak_sigma:.1f} m",
        f"analytic peak Pc           {curve.analytic_peak_probability:.6e} "
        f"(circular covariance reference)",
        f"analytic peak sigma        {curve.analytic_peak_sigma_m:.1f} m",
        f"dilution factor            {curve.dilution_factor:.2f}",
        f"in dilution region         {curve.in_dilution_region}",
        f"falling branch log slope   {curve.falling_branch_slope:.4f}",
    ]
    return "\n".join(lines)
