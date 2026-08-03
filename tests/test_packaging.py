"""Tests about what the installed distribution contains rather than what it computes.

A library can pass mypy in strict mode and still deliver no types at all to
anything that installs it. PEP 561 makes the marker file the switch: without
``py.typed`` inside the package directory, a type checker running over a
downstream project treats every import from here as ``Any`` and every annotation
written in this repository stops at its own boundary.

The tracked figures are checked in the same place, because they are the other
thing that is part of the published repository rather than part of the running
code.
"""

from __future__ import annotations

from pathlib import Path

import conjunction_screening

_PACKAGE = Path(conjunction_screening.__file__).resolve().parent
_REPOSITORY = Path(__file__).resolve().parent.parent
_FIGURES = _REPOSITORY / "docs" / "figures"
_FIGURE_BUDGET_BYTES = 250 * 1024


def test_the_package_ships_a_py_typed_marker() -> None:
    """The marker is inside the package directory, which is where PEP 561 looks."""
    marker = _PACKAGE / "py.typed"
    assert marker.is_file(), f"py.typed is missing from {_PACKAGE}"
    assert marker.parent.name == "conjunction_screening"
    assert (marker.parent / "__init__.py").is_file()


def test_the_marker_is_empty() -> None:
    """PEP 561 gives the file no contents; anything in it would be ignored."""
    assert (_PACKAGE / "py.typed").read_bytes() == b""


def test_the_tracked_figures_are_present_and_within_budget() -> None:
    """The README points at these files, and a repository is not a place for large ones.

    The figures are committed rather than generated on clone, so they have to be
    small enough that the repository stays cheap to fetch. The budget is the one
    the portfolio checker applies.
    """
    figures = sorted(_FIGURES.glob("*.png"))
    assert figures, f"no figure is tracked in {_FIGURES}"
    for figure in figures:
        assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{figure.name} is not a PNG"
    total = sum(figure.stat().st_size for figure in figures)
    assert total <= _FIGURE_BUDGET_BYTES, f"tracked figures total {total} bytes"


def test_every_tracked_figure_is_referenced_by_the_readme() -> None:
    """A figure nobody links to is weight without a reader."""
    readme = (_REPOSITORY / "README.md").read_text(encoding="utf-8")
    for figure in sorted(_FIGURES.glob("*.png")):
        target = f"docs/figures/{figure.name}"
        assert target in readme, f"{target} is tracked but not referenced by the README"
