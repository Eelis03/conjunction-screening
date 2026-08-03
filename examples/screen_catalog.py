"""Screen a synthetic catalogue and print the ranked conjunction report.

Wiring only. Every number printed comes from the library.

    uv run python examples/screen_catalog.py
    uv run python examples/screen_catalog.py --reduced
"""

from __future__ import annotations

import argparse
from pathlib import Path

from conjunction_screening.analysis.figures import plot_screening_scatter
from conjunction_screening.analysis.ranking import (
    ActionClass,
    ActionThresholds,
    format_covariance_table,
    format_ranking_table,
    rank_report,
)
from conjunction_screening.pipeline.catalog import generate_catalog
from conjunction_screening.pipeline.screening import ScreeningConfig, run_screening


def parse_arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=240, help="catalogue size")
    parser.add_argument("--planted", type=int, default=8, help="planted conjunctions")
    parser.add_argument("--window", type=float, default=86_400.0, help="screening window in s")
    parser.add_argument("--threshold", type=float, default=5_000.0, help="threshold in m")
    parser.add_argument("--seed", type=int, default=20260731, help="catalogue seed")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="directory for figures"
    )
    parser.add_argument(
        "--reduced", action="store_true", help="run a small catalogue for a fast check"
    )
    return parser.parse_args()


def main() -> None:
    """Generate a catalogue, screen it, and report the ranking."""
    arguments = parse_arguments()
    objects = 40 if arguments.reduced else arguments.objects
    planted = 4 if arguments.reduced else arguments.planted

    catalog = generate_catalog(
        count=objects, planted=planted, window_s=arguments.window, seed=arguments.seed
    )
    config = ScreeningConfig.for_threshold(arguments.threshold, arguments.window)
    report = run_screening(catalog, config)
    thresholds = ActionThresholds()
    ranked = rank_report(report, thresholds)

    print(f"catalogue        {catalog.size} secondaries, seed {catalog.seed}")
    print(f"window           {report.window_s:.0f} s")
    print(f"threshold        {report.threshold_m:.0f} m")
    print(f"probability      {report.method}")
    print()
    print("filter cascade")
    total = report.screened
    remaining = total
    for name in ("perigee-apogee", "orbit-path", "time"):
        rejected = report.rejection_counts.get(name, 0)
        remaining -= rejected
        print(f"  {name:<16} rejected {rejected:>4} of {total:>4}, {remaining:>4} remain")
    print(f"  {'survivors':<16} {report.survivors:>4}")
    print(f"  {'events':<16} {len(report.events):>4}")
    print(
        f"  {'candidate time':<16} {report.cascade_cost_windows:.1f} s of "
        f"{report.window_s * report.survivors:.0f} s that a window-wide search would cover"
    )
    print()
    print("ranked conjunctions")
    print(format_ranking_table(ranked, limit=15))
    print()
    print("covariance geometry, the miss distance in metres and in standard deviations")
    print(format_covariance_table(ranked))
    print()
    counts = {action.value: 0 for action in ActionClass}
    for item in ranked:
        counts[item.action.value] += 1
    print("action counts   " + ", ".join(f"{key} {value}" for key, value in counts.items()))

    if ranked:
        figure = plot_screening_scatter(
            ranked, thresholds, arguments.output / "screening_scatter.png"
        )
        print(f"figure written  {figure}")


if __name__ == "__main__":
    main()
