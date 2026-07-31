"""Figure generation for the analysis outputs.

The non-interactive Agg backend is selected on import so that a script writes
files without needing a display, which is what the continuous integration run
requires. This is the only module in the library that writes to disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from conjunction_screening.analysis.comparison import MethodComparison
from conjunction_screening.analysis.dilution import DilutionCurve
from conjunction_screening.analysis.ranking import (
    ActionThresholds,
    RankedConjunction,
)

__all__ = [
    "plot_dilution_curve",
    "plot_method_comparison",
    "plot_screening_ranking",
]


def plot_dilution_curve(
    curve: DilutionCurve, destination: Path, title: str | None = None
) -> Path:
    """Plot probability against covariance scale with the maximum marked."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7.5, 4.8))
    positive = curve.probabilities > 0.0
    axes.loglog(curve.scales[positive], curve.probabilities[positive], linewidth=1.8, label="Pc")
    axes.axvline(1.0, color="0.55", linestyle=":", linewidth=1.2, label="nominal covariance")
    axes.plot(
        [curve.peak.scale],
        [curve.peak.probability],
        marker="o",
        markersize=7,
        linestyle="none",
        color="crimson",
        label=(
            f"maximum Pc = {curve.peak.probability:.3e}\nat scale {curve.peak.scale:.3f}"
        ),
    )
    axes.axhline(
        curve.analytic_peak_probability,
        color="darkgreen",
        linestyle="--",
        linewidth=1.1,
        label="R^2 / (e d^2), the circular covariance maximum",
    )
    axes.set_xlabel("covariance scale factor applied to every standard deviation")
    axes.set_ylabel("probability of collision")
    axes.set_title(
        title
        or (
            f"Dilution of Pc, miss {curve.miss_distance_m:.0f} m, "
            f"hard body radius {curve.hard_body_radius_m:.1f} m"
        )
    )
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def plot_method_comparison(
    comparisons: tuple[MethodComparison, ...],
    method_order: tuple[str, ...],
    destination: Path,
) -> Path:
    """Plot each method's probability against the Foster value on a logarithmic scale."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = method_order[0]
    figure, axes = plt.subplots(figsize=(7.0, 5.4))
    baseline = np.array(
        [comparison.results[reference].value for comparison in comparisons], dtype=np.float64
    )
    markers = ["o", "s", "^", "D", "v"]
    for index, name in enumerate(method_order[1:], start=1):
        values = np.array(
            [comparison.results[name].value for comparison in comparisons], dtype=np.float64
        )
        axes.loglog(
            baseline,
            values,
            linestyle="none",
            marker=markers[index % len(markers)],
            markersize=6,
            alpha=0.8,
            label=name,
        )
    finite = baseline[baseline > 0.0]
    if finite.size:
        span = np.array([finite.min() * 0.5, finite.max() * 2.0])
        axes.loglog(span, span, color="0.4", linewidth=1.0, linestyle="--", label="equality")
    axes.set_xlabel(f"probability from {reference}")
    axes.set_ylabel("probability from the compared method")
    axes.set_title("Agreement between the probability of collision methods")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def plot_screening_ranking(
    ranked: tuple[RankedConjunction, ...],
    thresholds: ActionThresholds,
    destination: Path,
    limit: int = 20,
) -> Path:
    """Plot the ranked probabilities with the action thresholds drawn on."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shown = [item for item in ranked[:limit] if item.probability > 0.0]
    figure, axes = plt.subplots(figsize=(8.0, 4.8))
    if shown:
        colours = {"act": "crimson", "monitor": "darkorange", "dismiss": "steelblue"}
        axes.bar(
            [item.rank for item in shown],
            [item.probability for item in shown],
            color=[colours[item.action.value] for item in shown],
        )
        axes.set_yscale("log")
    axes.axhline(
        thresholds.act, color="crimson", linestyle="--", linewidth=1.2, label="act threshold"
    )
    axes.axhline(
        thresholds.monitor,
        color="darkorange",
        linestyle=":",
        linewidth=1.2,
        label="monitor threshold",
    )
    axes.set_xlabel("rank")
    axes.set_ylabel("probability of collision")
    axes.set_title(f"Screening report, top {len(shown)} conjunctions by probability")
    axes.grid(True, axis="y", which="both", alpha=0.3)
    axes.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination
