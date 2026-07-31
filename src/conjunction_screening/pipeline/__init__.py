"""Pipeline layer: synthetic catalogue generation and the screening run.

The pipeline wires the model and algorithm layers together and produces a
structured trace. It does not plot, rank, or format; those belong to the analysis
layer.
"""

from __future__ import annotations

from conjunction_screening.pipeline.catalog import (
    CatalogObject,
    PlantedConjunction,
    SyntheticCatalog,
    default_primary,
    generate_catalog,
    ric_covariance_from_elements,
)
from conjunction_screening.pipeline.screening import (
    ConjunctionEvent,
    PairTrace,
    ScreeningConfig,
    ScreeningReport,
    build_encounter,
    propagated_position_covariance,
    run_screening,
)

__all__ = [
    "CatalogObject",
    "ConjunctionEvent",
    "PairTrace",
    "PlantedConjunction",
    "ScreeningConfig",
    "ScreeningReport",
    "SyntheticCatalog",
    "build_encounter",
    "default_primary",
    "generate_catalog",
    "propagated_position_covariance",
    "ric_covariance_from_elements",
    "run_screening",
]
