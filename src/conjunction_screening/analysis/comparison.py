"""Cross comparison of the probability methods.

Foster's polar quadrature and Alfano's error function reduction evaluate the same
integral by different routes, so agreement between them checks both
implementations rather than restating one. Chan's series is an independent
analytic construction that is exact for a circular covariance and approximate
otherwise, which makes the size of its disagreement informative in itself. Monte
Carlo sampling checks the encounter plane construction as well as the integral,
at the cost of a standard error that has to be quoted with the value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conjunction_screening.algorithm.probability import ProbabilityMethod, ProbabilityResult
from conjunction_screening.model.encounter import EncounterGeometry, principal_axis_form

__all__ = [
    "MethodComparison",
    "compare_methods",
    "format_comparison_table",
    "relative_disagreement",
]


@dataclass(frozen=True, slots=True)
class MethodComparison:
    """Every method's answer for one encounter.

    Attributes:
        label: Identifier of the encounter.
        miss_distance_m: Projected miss distance, in m.
        hard_body_radius_m: Combined hard body radius, in m.
        sigma_x_m: Larger principal in-plane standard deviation, in m.
        sigma_y_m: Smaller principal in-plane standard deviation, in m.
        results: Result from each method, keyed by method name.
    """

    label: str
    miss_distance_m: float
    hard_body_radius_m: float
    sigma_x_m: float
    sigma_y_m: float
    results: dict[str, ProbabilityResult]

    @property
    def aspect_ratio(self) -> float:
        """Ratio of the two principal standard deviations, at least one."""
        return self.sigma_x_m / self.sigma_y_m


def relative_disagreement(first: float, second: float) -> float:
    """Return the absolute difference divided by the larger magnitude.

    Returns zero when both values are zero, which is agreement rather than an
    undefined ratio.
    """
    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return 0.0
    return abs(first - second) / scale


def compare_methods(
    encounters: dict[str, EncounterGeometry], methods: tuple[ProbabilityMethod, ...]
) -> tuple[MethodComparison, ...]:
    """Evaluate every method on every encounter."""
    comparisons: list[MethodComparison] = []
    for label, encounter in encounters.items():
        form = principal_axis_form(encounter)
        comparisons.append(
            MethodComparison(
                label=label,
                miss_distance_m=encounter.projected_miss_distance_m,
                hard_body_radius_m=encounter.hard_body_radius_m,
                sigma_x_m=form.sigma_x_m,
                sigma_y_m=form.sigma_y_m,
                results={method.name: method.probability(encounter) for method in methods},
            )
        )
    return tuple(comparisons)


def format_comparison_table(
    comparisons: tuple[MethodComparison, ...], method_order: tuple[str, ...]
) -> str:
    """Render a comparison as a fixed-width text table."""
    columns = "".join(f"  {name:>13}" for name in method_order)
    header = (
        f"{'case':<22}  {'miss [m]':>9}  {'R [m]':>6}  {'sx [m]':>8}  {'sy [m]':>8}{columns}"
        f"  {'max rel diff':>12}"
    )
    lines = [header, "-" * len(header)]
    for comparison in comparisons:
        values = [comparison.results[name].value for name in method_order]
        worst = 0.0
        for index, first in enumerate(values):
            for second in values[index + 1 :]:
                worst = max(worst, relative_disagreement(first, second))
        cells = "".join(f"  {value:>13.6e}" for value in values)
        lines.append(
            f"{comparison.label:<22}  {comparison.miss_distance_m:>9.1f}  "
            f"{comparison.hard_body_radius_m:>6.1f}  {comparison.sigma_x_m:>8.1f}  "
            f"{comparison.sigma_y_m:>8.1f}{cells}  {worst:>12.3e}"
        )
    return "\n".join(lines)


def worst_pairwise_disagreement(
    comparisons: tuple[MethodComparison, ...], first: str, second: str
) -> float:
    """Return the largest relative disagreement between two named methods."""
    return float(
        np.max(
            [
                relative_disagreement(
                    comparison.results[first].value, comparison.results[second].value
                )
                for comparison in comparisons
            ]
        )
    )
