"""Regenerate the three figures the README embeds.

This is the one command that rewrites docs/figures. Everything it draws comes
from the same seeded runs the other examples print, so a figure and the table
beside it in the README describe the same numbers. Wiring only.

    uv run python examples/render_figures.py
    uv run python examples/render_figures.py --reduced

Matplotlib does not produce byte identical output across platforms or across its
own patch releases, so the committed files are snapshots rather than build
artefacts, and nothing compares them byte for byte.
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
from conjunction_screening.analysis.comparison import compare_methods
from conjunction_screening.analysis.dilution import dilution_curve
from conjunction_screening.analysis.figures import (
    plot_dilution_curve,
    plot_method_agreement,
    plot_screening_scatter,
)
from conjunction_screening.analysis.ranking import ActionThresholds, rank_report
from conjunction_screening.model.encounter import EncounterGeometry, planar_encounter
from conjunction_screening.pipeline.catalog import generate_catalog
from conjunction_screening.pipeline.screening import ScreeningConfig, run_screening

# The same eight encounters the method comparison example tabulates: miss
# distance, sigma x, sigma y, hard body radius, and orientation in degrees.
_CASES: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("circular-near", 50.0, 100.0, 100.0, 10.0, 0.0),
    ("circular-mid", 200.0, 250.0, 250.0, 12.0, 0.0),
    ("circular-far", 700.0, 300.0, 300.0, 8.0, 0.0),
    ("elongated-2to1", 150.0, 400.0, 200.0, 10.0, 0.0),
    ("elongated-5to1", 300.0, 1_000.0, 200.0, 12.0, 30.0),
    ("elongated-20to1", 200.0, 2_000.0, 100.0, 15.0, 60.0),
    ("wide-covariance", 120.0, 3_000.0, 1_500.0, 10.0, 15.0),
    ("tight-covariance", 80.0, 60.0, 40.0, 9.0, 45.0),
)


def parse_arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=240, help="catalogue size")
    parser.add_argument("--planted", type=int, default=8, help="planted conjunctions")
    parser.add_argument("--seed", type=int, default=20260731, help="catalogue and sampling seed")
    parser.add_argument("--points", type=int, default=121, help="samples on the dilution curve")
    parser.add_argument("--samples", type=int, default=4_000_000, help="Monte Carlo draws")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "figures",
        help="directory the figures are written to",
    )
    parser.add_argument(
        "--reduced", action="store_true", help="draw from a small run, for a fast check"
    )
    return parser.parse_args()


def main() -> None:
    """Draw the dilution curve, the screening scatter, and the agreement plot."""
    arguments = parse_arguments()
    objects = 40 if arguments.reduced else arguments.objects
    planted = 4 if arguments.reduced else arguments.planted
    points = 25 if arguments.reduced else arguments.points
    samples = 100_000 if arguments.reduced else arguments.samples
    foster = FosterMethod()

    reference = planar_encounter(
        miss_distance_m=100.0, sigma_x_m=500.0, sigma_y_m=500.0, hard_body_radius_m=10.0
    )
    curve = dilution_curve(reference, foster, minimum_scale=1e-2, maximum_scale=1e3, points=points)
    dilution = plot_dilution_curve(
        curve,
        arguments.output / "dilution_curve.png",
        title=("Probability of collision against covariance scale, miss distance 100 m throughout"),
    )

    catalog = generate_catalog(
        count=objects, planted=planted, window_s=86_400.0, seed=arguments.seed
    )
    report = run_screening(catalog, ScreeningConfig.for_threshold(5_000.0))
    thresholds = ActionThresholds()
    scatter = plot_screening_scatter(
        rank_report(report, thresholds),
        thresholds,
        arguments.output / "screening_scatter.png",
    )

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
        foster,
        AlfanoMethod(),
        ChanMethod(),
        MonteCarloMethod(samples=samples, seed=arguments.seed),
    )
    agreement = plot_method_agreement(
        compare_methods(encounters, methods),
        (FOSTER, ALFANO, CHAN, MONTE_CARLO),
        arguments.output / "method_agreement.png",
    )

    print(f"dilution curve      {dilution}")
    print(f"screening scatter   {scatter}")
    print(f"method agreement    {agreement}")
    total = sum(path.stat().st_size for path in (dilution, scatter, agreement))
    print(f"total size          {total} bytes")


if __name__ == "__main__":
    main()
