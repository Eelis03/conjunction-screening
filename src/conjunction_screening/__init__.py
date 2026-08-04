"""Conjunction filtering and probability of collision using the Foster and Alfano methods.

The package is arranged in five layers, each depending only on the ones above it:

* ``model``: orbit states, covariances, frame rotations, and encounter plane
  geometry, all as pure functions over immutable values;
* ``algorithm``: the filter cascade, close approach refinement, and the
  probability methods behind one Protocol;
* ``pipeline``: synthetic catalogue generation and the screening run that
  produces a structured trace;
* ``analysis``: ranking, the dilution study, method comparison, and figures;
* ``examples``: thin wiring scripts outside the package.
"""

from __future__ import annotations

from conjunction_screening.algorithm import (
    AlfanoMethod,
    CascadeSettings,
    ChanMethod,
    CloseApproachSettings,
    FosterMethod,
    MonteCarloMethod,
    PateraMethod,
    ProbabilityMethod,
    ProbabilityResult,
    find_close_approaches,
    maximum_probability,
    run_cascade,
)
from conjunction_screening.model import (
    Covariance,
    EncounterGeometry,
    KeplerianElements,
    OrbitState,
)
from conjunction_screening.pipeline import (
    ScreeningConfig,
    ScreeningReport,
    generate_catalog,
    run_screening,
)

__all__ = [
    "AlfanoMethod",
    "CascadeSettings",
    "ChanMethod",
    "CloseApproachSettings",
    "Covariance",
    "EncounterGeometry",
    "FosterMethod",
    "KeplerianElements",
    "MonteCarloMethod",
    "OrbitState",
    "PateraMethod",
    "ProbabilityMethod",
    "ProbabilityResult",
    "ScreeningConfig",
    "ScreeningReport",
    "__version__",
    "find_close_approaches",
    "generate_catalog",
    "maximum_probability",
    "run_cascade",
    "run_screening",
]

__version__ = "0.1.0"
