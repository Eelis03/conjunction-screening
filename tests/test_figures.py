"""Tests for the figure module.

A figure cannot be asserted to be readable, and a test that only checked a file
appeared would pass on a blank canvas. What can be checked is that each function
writes a real PNG and that the drawing decisions the module argues for are the
ones it makes: the vertical window of the dilution curve is set from the peak
rather than from the data, every screened event reaches the canvas even when its
probability is hundreds of decades below the floor, and a cross check that agrees
bit for bit still plots on a logarithmic axis.

The axes are inspected by intercepting the call that creates them, so the
assertions are about the artists that were drawn rather than about a recomputed
copy of the same formula.

These run in process, unlike the example scripts, which run as subprocesses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
from conjunction_screening.analysis import figures as figures_module
from conjunction_screening.analysis.comparison import compare_methods
from conjunction_screening.analysis.dilution import DilutionCurve, dilution_curve
from conjunction_screening.analysis.figures import (
    FIGURE_DPI,
    plot_dilution_curve,
    plot_method_agreement,
    plot_screening_scatter,
)
from conjunction_screening.analysis.ranking import ActionThresholds, rank_report
from conjunction_screening.model.encounter import EncounterGeometry, planar_encounter
from conjunction_screening.pipeline.screening import ScreeningReport

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def drawn_axes(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture the axes each plotting call creates, after the call has closed them."""
    captured: list[Any] = []
    original = plt.subplots

    def spy(*args: Any, **kwargs: Any) -> Any:
        figure, axes = original(*args, **kwargs)
        captured.append(axes)
        return figure, axes

    monkeypatch.setattr(figures_module.plt, "subplots", spy)
    return captured


def _reference_curve() -> DilutionCurve:
    encounter = planar_encounter(
        miss_distance_m=100.0, sigma_x_m=500.0, sigma_y_m=500.0, hard_body_radius_m=10.0
    )
    return dilution_curve(
        encounter, FosterMethod(), minimum_scale=1e-2, maximum_scale=1e3, points=25
    )


def test_the_dilution_figure_is_a_png_in_a_created_directory(tmp_path: Path) -> None:
    """The function makes its own output directory and writes a real PNG."""
    destination = tmp_path / "nested" / "dilution.png"
    written = plot_dilution_curve(_reference_curve(), destination)
    assert written == destination
    assert destination.read_bytes()[:8] == _PNG_MAGIC
    assert destination.stat().st_size > 5_000


def test_the_dilution_figure_frames_the_peak_rather_than_the_data(
    tmp_path: Path, drawn_axes: list[Any]
) -> None:
    """The vertical window is set from the peak, so the hump is visible.

    The rising branch of this curve reaches 1e-73 at the smallest scale sampled.
    Letting the data set the limits would spend seventy decades of the axis on
    values indistinguishable from zero and compress the entire finding into a few
    pixels, which is the failure this choice exists to prevent.
    """
    curve = _reference_curve()
    assert float(np.min(curve.probabilities)) < 1e-40

    plot_dilution_curve(curve, tmp_path / "dilution.png", floor_decades=6.0)
    lower, upper = drawn_axes[0].get_ylim()
    assert lower == pytest.approx(curve.peak.probability * 1e-6, rel=1e-12)
    assert upper > curve.peak.probability
    assert lower > float(np.min(curve.probabilities))


def test_the_dilution_figure_marks_the_peak_and_the_nominal_covariance(
    tmp_path: Path, drawn_axes: list[Any]
) -> None:
    """Both points the caption talks about are drawn where the numbers say."""
    curve = _reference_curve()
    plot_dilution_curve(curve, tmp_path / "dilution.png")
    marked = {
        (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
        for line in drawn_axes[0].get_lines()
        if len(line.get_xdata()) == 1
    }
    assert (curve.peak.scale, curve.peak.probability) in marked
    assert (1.0, curve.nominal_probability) in marked


def test_the_screening_figure_puts_every_event_on_the_canvas(
    regression_report: ScreeningReport, tmp_path: Path, drawn_axes: list[Any]
) -> None:
    """Events below the floor are drawn on it rather than dropped.

    A screening run produces probabilities hundreds of decades apart. Dropping
    the ones below the floor would quietly remove conjunctions from a figure
    whose whole subject is how they compare.
    """
    ranked = rank_report(regression_report, ActionThresholds())
    assert min(item.probability for item in ranked) < 1e-20

    destination = tmp_path / "scatter.png"
    written = plot_screening_scatter(ranked, ActionThresholds(), destination, floor=1e-20)
    assert written.read_bytes()[:8] == _PNG_MAGIC

    points = [line for line in drawn_axes[0].get_lines() if len(line.get_xdata()) == 1]
    assert len(points) == len(ranked)
    assert {float(line.get_xdata()[0]) for line in points} == {
        item.miss_distance_m for item in ranked
    }
    assert all(float(line.get_ydata()[0]) >= 1e-20 for line in points)
    assert len(drawn_axes[0].texts) == len(ranked)


def test_the_agreement_figure_handles_a_method_that_matches_exactly(
    tmp_path: Path, drawn_axes: list[Any]
) -> None:
    """A relative difference of zero cannot be plotted on a logarithmic axis.

    Foster and Alfano reach the same double on some cases, and a figure that
    raised or silently dropped those points would be hiding its best result.
    """
    encounters: dict[str, EncounterGeometry] = {
        "circular": planar_encounter(
            miss_distance_m=50.0, sigma_x_m=100.0, sigma_y_m=100.0, hard_body_radius_m=10.0
        ),
        "elongated": planar_encounter(
            miss_distance_m=300.0, sigma_x_m=1_000.0, sigma_y_m=200.0, hard_body_radius_m=12.0
        ),
    }
    methods: tuple[ProbabilityMethod, ...] = (
        FosterMethod(),
        AlfanoMethod(),
        ChanMethod(),
        MonteCarloMethod(samples=200_000, seed=4242),
    )
    comparisons = compare_methods(encounters, methods)
    exact = [
        comparison
        for comparison in comparisons
        if comparison.results[FOSTER].value == comparison.results[ALFANO].value
    ]
    assert exact, "expected a case where the two quadratures reach the same double"

    destination = tmp_path / "agreement.png"
    written = plot_method_agreement(comparisons, (FOSTER, ALFANO, CHAN, MONTE_CARLO), destination)
    assert written.read_bytes()[:8] == _PNG_MAGIC

    plotted = np.concatenate(
        [np.asarray(line.get_ydata(), dtype=np.float64) for line in drawn_axes[0].get_lines()]
    )
    assert plotted.size > 0
    assert bool(np.all(plotted > 0.0)), "a zero would vanish from a logarithmic axis"


def test_the_resolution_matches_the_width_a_page_renders(tmp_path: Path) -> None:
    """The pixel width is a deliberate number, not a matplotlib default.

    A Markdown page renders an embedded image at about 880 pixels. The figures
    are drawn 7.4 inches wide, so the resolution has to put the result near that:
    smaller looks soft, larger spends bytes on pixels nobody sees.
    """
    destination = tmp_path / "dilution.png"
    plot_dilution_curve(_reference_curve(), destination)
    width = int.from_bytes(destination.read_bytes()[16:20], "big")
    assert 860 <= width <= 900
    assert FIGURE_DPI == 120
