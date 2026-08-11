"""Figure generation for the analysis outputs.

The non-interactive Agg backend is selected on import so that a script writes
files without needing a display, which is what the continuous integration run
requires. This is the only module in the library that writes to disk.

Three figures are drawn, and each exists because a number cannot make the same
point. The dilution curve shows a probability that rises and then falls while the
miss distance never moves. The screening scatter shows that ordering conjunctions
by miss distance is not ordering them by risk. The agreement plot shows three
cross checks sitting at three different precision floors, from the machine
epsilon of two quadratures to the sampling noise of a Monte Carlo run.

Every axis that carries a probability spans many decades, and a plot that lets
the full range set its own limits shows a flat line and a cliff. The limits are
therefore chosen from the quantity being explained rather than from the data
extent, and anything outside them is drawn on the boundary with a marker that
says it is off the scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

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
    "FIGURE_DPI",
    "plot_dilution_curve",
    "plot_method_agreement",
    "plot_screening_scatter",
]

FIGURE_DPI: Final[int] = 120
"""Resolution every figure is written at.

A Markdown page renders an embedded image at about 880 pixels wide, so a 7.4 inch
figure at 120 dpi is 888 pixels: sharp at the width it is shown at and no larger.
Doubling the resolution would roughly double the file size to buy detail that the
page never displays, and the three tracked figures share a 250 KB budget.
"""

_DILUTION_SIZE: Final[tuple[float, float]] = (7.4, 4.3)
_SCATTER_SIZE: Final[tuple[float, float]] = (7.4, 4.4)
_AGREEMENT_SIZE: Final[tuple[float, float]] = (7.4, 4.6)

_ACTION_COLOURS: Final[dict[str, str]] = {
    "act": "crimson",
    "monitor": "darkorange",
    "dismiss": "steelblue",
}


def plot_dilution_curve(
    curve: DilutionCurve,
    destination: Path,
    title: str | None = None,
    floor_decades: float = 6.0,
    dpi: int = FIGURE_DPI,
) -> Path:
    """Plot probability against covariance scale with the dilution region marked.

    Args:
        curve: The swept curve, its peak, and its analytic reference.
        destination: File to write. Parent directories are created.
        title: Overrides the generated title.
        floor_decades: Decades below the peak shown on the vertical axis. The
            rising branch falls through the floor of any window, because the
            probability of a fixed miss distance under a vanishing covariance
            goes to zero faster than exponentially; six decades is enough to see
            the rise, the peak, and the whole falling branch.
        dpi: Output resolution.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=_DILUTION_SIZE)
    positive = curve.probabilities > 0.0
    peak = curve.peak.probability

    axes.axvspan(
        curve.peak.scale,
        float(curve.scales[-1]),
        color="0.88",
        alpha=0.55,
        label="dilution region, wider covariance and smaller Pc",
    )
    axes.loglog(curve.scales[positive], curve.probabilities[positive], linewidth=2.0, label="Pc")
    axes.axhline(
        curve.analytic_peak_probability,
        color="darkgreen",
        linestyle="--",
        linewidth=1.1,
        label="R^2 / (e d^2), the circular covariance maximum",
    )
    axes.plot(
        [curve.peak.scale],
        [peak],
        marker="o",
        markersize=7.5,
        linestyle="none",
        color="crimson",
        label=f"maximum Pc {peak:.3e} at scale {curve.peak.scale:.3f}",
    )
    axes.plot(
        [1.0],
        [curve.nominal_probability],
        marker="D",
        markersize=6.5,
        linestyle="none",
        color="black",
        label=(
            f"nominal covariance, Pc {curve.nominal_probability:.3e}, "
            f"{curve.dilution_factor:.1f} times below the maximum"
        ),
    )

    axes.set_ylim(peak * 10.0**-floor_decades, peak * 6.0)
    axes.set_xlabel("covariance scale factor applied to every standard deviation")
    axes.set_ylabel("probability of collision")
    axes.set_title(
        title
        or (
            f"Pc against covariance scale, miss {curve.miss_distance_m:.0f} m held fixed, "
            f"hard body radius {curve.hard_body_radius_m:.1f} m"
        ),
        fontsize=11,
    )
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(loc="upper right", fontsize=8, framealpha=0.93)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def plot_screening_scatter(
    ranked: tuple[RankedConjunction, ...],
    thresholds: ActionThresholds,
    destination: Path,
    floor: float = 1e-20,
    dpi: int = FIGURE_DPI,
) -> Path:
    """Plot each screened conjunction as probability against miss distance.

    The two quantities disagree about the ordering, which is the point of the
    figure: a conjunction further away can carry the higher probability when its
    combined covariance is tighter.

    Args:
        ranked: The ranked conjunctions of one screening run.
        thresholds: Action thresholds, drawn as horizontal lines.
        destination: File to write.
        floor: Probability below which an event is drawn on the bottom edge with
            a downward marker. Values several hundred decades below the
            dismissal threshold carry no decision information, and letting them
            set the axis would compress everything that does.
        dpi: Output resolution.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=_SCATTER_SIZE)

    seen: set[str] = set()
    for item in ranked:
        below = item.probability < floor
        colour = _ACTION_COLOURS[item.action.value]
        label = None if item.action.value in seen else item.action.value
        seen.add(item.action.value)
        axes.plot(
            [item.miss_distance_m],
            [floor if below else item.probability],
            marker="v" if below else "o",
            markersize=9 if below else 8,
            linestyle="none",
            color=colour,
            markeredgecolor="0.25",
            label=label,
        )
        axes.annotate(
            item.object_id.replace("PLANTED-", "P"),
            (item.miss_distance_m, floor if below else item.probability),
            textcoords="offset points",
            # Two events of nearly equal probability would otherwise print their
            # labels on top of each other, so successive ranks alternate sides.
            xytext=(9, 4) if item.rank % 2 == 0 else (9, -10),
            fontsize=8,
            color="0.25",
        )

    axes.axhline(
        thresholds.act, color="crimson", linestyle="--", linewidth=1.1, label="act threshold"
    )
    axes.axhline(
        thresholds.monitor,
        color="darkorange",
        linestyle=":",
        linewidth=1.2,
        label="monitor threshold",
    )
    axes.set_yscale("log")
    axes.set_ylim(floor / 6.0, max(thresholds.act * 40.0, 1e-3))
    axes.set_xlabel("miss distance at the time of closest approach [m]")
    axes.set_ylabel(f"probability of collision, floor at {floor:.0e}")
    axes.set_title("Screened conjunctions: miss distance does not order the risk", fontsize=11)
    axes.grid(True, which="major", alpha=0.3)
    axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def plot_method_agreement(
    comparisons: tuple[MethodComparison, ...],
    method_order: tuple[str, ...],
    destination: Path,
    monte_carlo: str = "monte-carlo",
    dpi: int = FIGURE_DPI,
) -> Path:
    """Plot each method's relative difference from the reference method.

    A plot of one method against another lies on the diagonal and shows nothing;
    the interesting quantity is how far off the diagonal each one is, and that
    spans fourteen decades. Drawing the difference itself puts each cross check
    at its own precision floor: two independent quadratures at machine epsilon, a
    series approximation at the error its own substitution introduces, and a
    Monte Carlo estimate at its sampling noise, which is drawn alongside so that
    its deviations can be read against it.

    Args:
        comparisons: One entry per encounter.
        method_order: Method names, the first of which is the reference.
        destination: File to write.
        monte_carlo: Name of the sampling method, whose standard error is drawn.
        dpi: Output resolution.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = method_order[0]
    figure, axes = plt.subplots(figsize=_AGREEMENT_SIZE)

    usable = [comparison for comparison in comparisons if comparison.results[reference].value > 0.0]
    baseline = np.array(
        [comparison.results[reference].value for comparison in usable], dtype=np.float64
    )
    floor = 1e-17
    markers = ["s", "^", "D", "v"]
    for index, name in enumerate(method_order[1:]):
        values = np.array(
            [comparison.results[name].value for comparison in usable], dtype=np.float64
        )
        difference = np.maximum(np.abs(values - baseline) / baseline, floor)
        axes.loglog(
            baseline,
            difference,
            linestyle="none",
            marker=markers[index % len(markers)],
            markersize=7,
            alpha=0.85,
            label=f"{name} against {reference}",
        )

    if monte_carlo in method_order:
        errors = np.array(
            [comparison.results[monte_carlo].error_estimate for comparison in usable],
            dtype=np.float64,
        )
        order = np.argsort(baseline)
        axes.loglog(
            baseline[order],
            (errors / baseline)[order],
            color="0.35",
            linewidth=1.2,
            linestyle="--",
            label=f"{monte_carlo} one sigma sampling noise",
        )

    axes.axhline(2.2e-16, color="0.6", linewidth=1.0, linestyle=":")
    axes.text(
        float(baseline.min()),
        5.0e-16,
        "double precision epsilon",
        fontsize=7.5,
        color="0.4",
        verticalalignment="bottom",
    )
    axes.set_ylim(floor / 2.0, 1.0)
    axes.set_xlabel(f"probability of collision from {reference}")
    axes.set_ylabel(f"relative difference from {reference}, floor at {floor:.0e}")
    axes.set_title("Three cross checks, each at its own precision floor", fontsize=11)
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(loc="center right", fontsize=8, framealpha=0.93)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination
