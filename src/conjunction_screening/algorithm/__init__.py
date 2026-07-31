"""Algorithm layer: the filter cascade, close approach refinement, and probability.

Nothing in this layer plots, prints, or reads a file. Every entry point takes
model values and returns model values or plain result records.
"""

from __future__ import annotations

from conjunction_screening.algorithm.close_approach import (
    CloseApproach,
    CloseApproachSettings,
    coarse_step_for,
    find_close_approaches,
    relative_state,
)
from conjunction_screening.algorithm.filters import (
    PATH_FILTER,
    PERIGEE_APOGEE_FILTER,
    TIME_FILTER,
    CascadeResult,
    CascadeSettings,
    FilterVerdict,
    PathSeparation,
    at_risk_arcs,
    minimum_path_separation,
    orbit_path_filter,
    perigee_apogee_filter,
    run_cascade,
    time_filter,
)
from conjunction_screening.algorithm.maximum import (
    MaximumProbability,
    isotropic_maximum_probability,
    isotropic_maximum_sigma_m,
    maximum_probability,
)
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
    ProbabilityResult,
)
from conjunction_screening.algorithm.propagation import (
    propagate,
    propagate_covariance,
    propagate_many,
    propagate_to,
    state_transition_matrix,
    symplectic_form,
    symplectic_residual,
)

__all__ = [
    "ALFANO",
    "CHAN",
    "FOSTER",
    "MONTE_CARLO",
    "PATH_FILTER",
    "PERIGEE_APOGEE_FILTER",
    "TIME_FILTER",
    "AlfanoMethod",
    "CascadeResult",
    "CascadeSettings",
    "ChanMethod",
    "CloseApproach",
    "CloseApproachSettings",
    "FilterVerdict",
    "FosterMethod",
    "MaximumProbability",
    "MonteCarloMethod",
    "PathSeparation",
    "ProbabilityMethod",
    "ProbabilityResult",
    "at_risk_arcs",
    "coarse_step_for",
    "find_close_approaches",
    "isotropic_maximum_probability",
    "isotropic_maximum_sigma_m",
    "maximum_probability",
    "minimum_path_separation",
    "orbit_path_filter",
    "perigee_apogee_filter",
    "propagate",
    "propagate_covariance",
    "propagate_many",
    "propagate_to",
    "relative_state",
    "run_cascade",
    "state_transition_matrix",
    "symplectic_form",
    "symplectic_residual",
    "time_filter",
]
