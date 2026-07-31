"""Compare the Foster, Alfano, Chan, and Monte Carlo probability methods.

A set of encounters with known plane geometry is built, every method is evaluated
on each, and the disagreements are tabulated. The Monte Carlo column carries its
own standard error, which is the only honest way to read it. Wiring only.

    uv run python examples/method_comparison.py
    uv run python examples/method_comparison.py --reduced
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
from conjunction_screening.analysis.comparison import (
    compare_methods,
    format_comparison_table,
    worst_pairwise_disagreement,
)
from conjunction_screening.analysis.figures import plot_method_comparison
from conjunction_screening.model.encounter import EncounterGeometry, planar_encounter

# miss distance, sigma x, sigma y, hard body radius, orientation in degrees
_CASES: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("circular-near", 50.0, 100.0, 100.0, 10.0, 0.0),
    ("circular-mid", 200.0, 250.0, 250.0, 12.0, 0.0),
    ("circular-far", 700.0, 300.0, 300.0, 8.0, 0.0),
    ("elongated-2to1", 150.0, 400.0, 200.0, 10.0, 0.0),
    ("elongated-5to1", 300.0, 1000.0, 200.0, 12.0, 30.0),
    ("elongated-20to1", 200.0, 2000.0, 100.0, 15.0, 60.0),
    ("wide-covariance", 120.0, 3000.0, 1500.0, 10.0, 15.0),
    ("tight-covariance", 80.0, 60.0, 40.0, 9.0, 45.0),
)


def parse_arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4_000_000, help="Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=20260731, help="Monte Carlo seed")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="directory for figures"
    )
    parser.add_argument(
        "--reduced", action="store_true", help="use fewer Monte Carlo draws for a fast check"
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate every method on every case and print the comparison."""
    arguments = parse_arguments()
    samples = 200_000 if arguments.reduced else arguments.samples

    encounters: dict[str, EncounterGeometry] = {
        label: planar_encounter(
            miss_distance_m=miss,
            sigma_x_m=sigma_x,
            sigma_y_m=sigma_y,
            hard_body_radius_m=radius,
            orientation_rad=orientation * 3.141592653589793 / 180.0,
        )
        for label, miss, sigma_x, sigma_y, radius, orientation in _CASES
    }
    methods: tuple[ProbabilityMethod, ...] = (
        FosterMethod(),
        AlfanoMethod(),
        ChanMethod(),
        MonteCarloMethod(samples=samples, seed=arguments.seed),
    )
    order = (FOSTER, ALFANO, CHAN, MONTE_CARLO)
    comparisons = compare_methods(encounters, methods)

    print(f"Monte Carlo draws per case  {samples}")
    print()
    print(format_comparison_table(comparisons, order))
    print()
    print(
        f"worst Foster against Alfano relative difference   "
        f"{worst_pairwise_disagreement(comparisons, FOSTER, ALFANO):.3e}"
    )
    print(
        f"worst Foster against Chan relative difference     "
        f"{worst_pairwise_disagreement(comparisons, FOSTER, CHAN):.3e}"
    )
    print()
    print("Monte Carlo cross check, deviation from Foster in units of its standard error")
    header = (
        f"{'case':<22}  {'foster':>13}  {'monte carlo':>13}  {'std error':>9}  "
        f"{'deviation':>9}  usable"
    )
    print(header)
    print("-" * len(header))
    for comparison in comparisons:
        analytic = comparison.results[FOSTER].value
        sampled = comparison.results[MONTE_CARLO]
        if sampled.converged and sampled.error_estimate > 0.0:
            deviation = f"{abs(sampled.value - analytic) / sampled.error_estimate:>9.2f}"
        else:
            deviation = f"{'n/a':>9}"
        print(
            f"{comparison.label:<22}  {analytic:>13.6e}  {sampled.value:>13.6e}  "
            f"{sampled.error_estimate:>9.3e}  {deviation}  {sampled.converged!s:<5}"
        )
    print()
    print("A case is usable when it produced at least the method's minimum hit count;")
    print("below that the binomial estimate carries no useful precision.")

    figure = plot_method_comparison(
        comparisons, order, arguments.output / "method_comparison.png"
    )
    print()
    print(f"figure written  {figure}")


if __name__ == "__main__":
    main()
