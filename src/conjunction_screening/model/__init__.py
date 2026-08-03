"""Model layer: orbit states, covariances, frames, and encounter plane geometry.

Every function in this layer is pure. Nothing reads a file, writes a figure, or
holds mutable state, so every result is reproducible from its arguments alone.
"""

from __future__ import annotations

from conjunction_screening.model.arrays import Matrix, Vector, as_matrix, as_vector, unit_vector
from conjunction_screening.model.constants import EARTH_RADIUS_M, J2_EARTH, MU_EARTH
from conjunction_screening.model.covariance import (
    Covariance,
    ElementSigmas,
    combine_covariances,
    covariance_from_element_sigmas,
    is_symmetric_positive_semidefinite,
    smallest_eigenvalue,
)
from conjunction_screening.model.encounter import (
    EncounterGeometry,
    PrincipalForm,
    encounter_plane_basis,
    planar_encounter,
    principal_axis_form,
    project_to_encounter_plane,
)
from conjunction_screening.model.frames import (
    block_diagonal_rotation,
    inertial_to_ric_rotation,
    rotate_covariance_to_inertial,
    rotate_covariance_to_ric,
)
from conjunction_screening.model.hardbody import (
    CrossSection,
    HardBody,
    combine_hard_bodies,
    projected_cross_section,
)
from conjunction_screening.model.state import (
    KeplerianElements,
    OrbitState,
    element_state_jacobian,
    elements_from_state,
    path_positions,
    solve_kepler_equation,
    state_from_elements,
    state_from_mean_elements,
)

__all__ = [
    "EARTH_RADIUS_M",
    "J2_EARTH",
    "MU_EARTH",
    "Covariance",
    "CrossSection",
    "ElementSigmas",
    "EncounterGeometry",
    "HardBody",
    "KeplerianElements",
    "Matrix",
    "OrbitState",
    "PrincipalForm",
    "Vector",
    "as_matrix",
    "as_vector",
    "block_diagonal_rotation",
    "combine_covariances",
    "combine_hard_bodies",
    "covariance_from_element_sigmas",
    "element_state_jacobian",
    "elements_from_state",
    "encounter_plane_basis",
    "inertial_to_ric_rotation",
    "is_symmetric_positive_semidefinite",
    "path_positions",
    "planar_encounter",
    "principal_axis_form",
    "project_to_encounter_plane",
    "projected_cross_section",
    "rotate_covariance_to_inertial",
    "rotate_covariance_to_ric",
    "smallest_eigenvalue",
    "solve_kepler_equation",
    "state_from_elements",
    "state_from_mean_elements",
    "unit_vector",
]
