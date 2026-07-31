"""Show that the probability of collision is not monotonic in covariance size.

Two cases are swept over five decades of covariance scale. The first is an
isotropic reference whose peak has a closed form, ``R^2 / (e d^2)`` at a standard
deviation of ``d / sqrt(2)``, so the numerical maximum can be checked against it.
The second is the highest-ranked conjunction from a screening run, whose in-plane
covariance is strongly elongated, for which the closed form is a comparison point
rather than a target, because it holds for circular covariances only. Wiring
only.

    uv run python examples/dilution_study.py
    uv run python examples/dilution_study.py --reduced
"""

from __future__ import annotations

import argparse
from pathlib import Path

from conjunction_screening.algorithm.probability import FosterMethod
from conjunction_screening.analysis.dilution import (
    DilutionCurve,
    dilution_curve,
    format_dilution_summary,
)
from conjunction_screening.analysis.figures import plot_dilution_curve
from conjunction_screening.model.encounter import planar_encounter
from conjunction_screening.pipeline.catalog import generate_catalog
from conjunction_screening.pipeline.screening import ScreeningConfig, run_screening


def parse_arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=240, help="catalogue size")
    parser.add_argument("--planted", type=int, default=8, help="planted conjunctions")
    parser.add_argument("--seed", type=int, default=20260731, help="catalogue seed")
    parser.add_argument("--points", type=int, default=121, help="samples on each curve")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="directory for figures"
    )
    parser.add_argument(
        "--reduced", action="store_true", help="run a small catalogue and a coarse curve"
    )
    return parser.parse_args()


def report_curve(name: str, curve: DilutionCurve, stride_target: int = 12) -> None:
    """Print one curve summary and a sample of its points."""
    print(name)
    print("-" * len(name))
    print(format_dilution_summary(curve))
    print()
    print("     scale             Pc")
    stride = max(len(curve.scales) // stride_target, 1)
    for scale, probability in zip(
        curve.scales[::stride], curve.probabilities[::stride], strict=True
    ):
        print(f"{scale:10.4f}   {probability:.6e}")
    print()


def main() -> None:
    """Sweep the covariance scale for an isotropic reference and a screened event."""
    arguments = parse_arguments()
    objects = 40 if arguments.reduced else arguments.objects
    planted = 4 if arguments.reduced else arguments.planted
    points = 25 if arguments.reduced else arguments.points
    method = FosterMethod()

    reference = planar_encounter(
        miss_distance_m=100.0, sigma_x_m=500.0, sigma_y_m=500.0, hard_body_radius_m=10.0
    )
    reference_curve = dilution_curve(
        reference, method, minimum_scale=1e-2, maximum_scale=1e3, points=points
    )
    report_curve("isotropic reference, miss 100 m, sigma 500 m, radius 10 m", reference_curve)

    catalog = generate_catalog(count=objects, planted=planted, seed=arguments.seed)
    report = run_screening(catalog, ScreeningConfig.for_threshold(5_000.0))
    if not report.events:
        raise SystemExit("no conjunction events were found; nothing to study")
    leading = report.events[0]
    screened_curve = dilution_curve(
        leading.encounter, method, minimum_scale=1e-2, maximum_scale=1e3, points=points
    )
    report_curve(
        f"screened conjunction {leading.object_id}, "
        f"time of closest approach {leading.tca_s:.3f} s",
        screened_curve,
    )

    first = plot_dilution_curve(
        reference_curve,
        arguments.output / "dilution_isotropic.png",
        title="Dilution of Pc, isotropic covariance, miss 100 m, radius 10 m",
    )
    second = plot_dilution_curve(
        screened_curve,
        arguments.output / "dilution_screened.png",
        title=(
            f"Dilution of Pc, {leading.object_id}, miss "
            f"{screened_curve.miss_distance_m:.0f} m, radius "
            f"{screened_curve.hard_body_radius_m:.1f} m"
        ),
    )
    print(f"figures written  {first}  {second}")


if __name__ == "__main__":
    main()
