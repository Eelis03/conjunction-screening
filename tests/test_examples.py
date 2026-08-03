"""Integration tests: every example script runs to completion.

The scripts are launched as subprocesses rather than imported, so the test covers
the command line handling and the module level matplotlib backend selection as
well as the code paths. Each runs with its reduced flag, which cuts the catalogue
size and the Monte Carlo sample count so the whole tier stays inside a few
seconds.

Every script is also given an output directory under the temporary path, which
matters most for the figure renderer: its default destination is the tracked
docs/figures, and a test run must not rewrite files that are committed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_SCRIPTS = (
    "screen_catalog.py",
    "dilution_study.py",
    "method_comparison.py",
    "render_figures.py",
)


@pytest.mark.parametrize("script", _SCRIPTS)
def test_example_runs_to_completion(script: str, tmp_path: Path) -> None:
    """The script exits cleanly, prints a report, and writes its figures."""
    path = _EXAMPLES / script
    assert path.exists(), f"{script} is missing from examples/"
    completed = subprocess.run(
        [sys.executable, str(path), "--reduced", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
    assert list(tmp_path.glob("*.png")), f"{script} wrote no figure"


@pytest.mark.parametrize("script", _SCRIPTS)
def test_example_offers_help(script: str) -> None:
    """Each script documents its own options."""
    completed = subprocess.run(
        [sys.executable, str(_EXAMPLES / script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--reduced" in completed.stdout


def test_screening_example_reports_the_cascade(tmp_path: Path) -> None:
    """The screening example prints the filter cascade and a ranked table."""
    completed = subprocess.run(
        [
            sys.executable,
            str(_EXAMPLES / "screen_catalog.py"),
            "--reduced",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for expected in ("filter cascade", "perigee-apogee", "orbit-path", "ranked conjunctions"):
        assert expected in completed.stdout


def test_comparison_example_reports_the_cross_checks(tmp_path: Path) -> None:
    """The comparison example prints both cross checks that the README quotes."""
    completed = subprocess.run(
        [
            sys.executable,
            str(_EXAMPLES / "method_comparison.py"),
            "--reduced",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "worst Foster against Alfano relative difference" in completed.stdout
    assert "standard error" in completed.stdout
