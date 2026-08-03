"""The screening run: catalogue in, ranked conjunction events and a trace out.

The run does one pass over the catalogue. For each secondary it applies the
filter cascade, searches the surviving time windows for close approaches, builds
the encounter geometry with a covariance propagated to the time of closest
approach, and evaluates the probability of collision. Every pair leaves a trace
entry whether or not it survives, so the cost and the effect of each filter can
be read off the report rather than inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from conjunction_screening.algorithm.close_approach import (
    CloseApproach,
    CloseApproachSettings,
    find_close_approaches,
)
from conjunction_screening.algorithm.filters import (
    CascadeResult,
    CascadeSettings,
    FilterVerdict,
    run_cascade,
)
from conjunction_screening.algorithm.probability import (
    FosterMethod,
    ProbabilityMethod,
    ProbabilityResult,
)
from conjunction_screening.algorithm.propagation import (
    propagate_covariance,
    propagate_to,
    state_transition_matrix,
)
from conjunction_screening.model.covariance import (
    Covariance,
    combine_covariances,
    is_symmetric_positive_semidefinite,
)
from conjunction_screening.model.encounter import (
    EncounterGeometry,
    encounter_plane_basis,
    project_to_encounter_plane,
)
from conjunction_screening.model.frames import (
    rotate_covariance_to_inertial,
    rotate_covariance_to_ric,
)
from conjunction_screening.model.hardbody import combine_hard_bodies, projected_cross_section
from conjunction_screening.model.state import OrbitState
from conjunction_screening.pipeline.catalog import CatalogObject, SyntheticCatalog

__all__ = [
    "ConjunctionEvent",
    "PairTrace",
    "ScreeningConfig",
    "ScreeningReport",
    "build_encounter",
    "propagated_position_covariance",
    "run_screening",
]

_DEFAULT_THRESHOLD_M: Final[float] = 5_000.0


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    """Configuration for one screening run.

    Attributes:
        cascade: Filter cascade tuning, which also carries the screening
            threshold and window.
        close_approach: Close approach search tuning.
    """

    cascade: CascadeSettings = field(default_factory=CascadeSettings)
    close_approach: CloseApproachSettings = field(
        default_factory=lambda: CloseApproachSettings(threshold_m=_DEFAULT_THRESHOLD_M)
    )

    @classmethod
    def for_threshold(cls, threshold_m: float, window_s: float = 86_400.0) -> ScreeningConfig:
        """Build a configuration with a matching threshold in both stages."""
        return cls(
            cascade=CascadeSettings(threshold_m=threshold_m, window_s=window_s),
            close_approach=CloseApproachSettings(threshold_m=threshold_m),
        )


@dataclass(frozen=True, slots=True)
class PairTrace:
    """What happened to one primary and secondary pair.

    Attributes:
        object_id: Secondary identifier.
        verdicts: Filter verdicts in application order.
        rejected_by: Name of the filter that rejected the pair, or None.
        candidate_windows: Number of candidate time windows the cascade produced.
        candidate_seconds: Total duration of those windows, in s.
        close_approaches: Number of close approaches found inside them.
    """

    object_id: str
    verdicts: tuple[FilterVerdict, ...]
    rejected_by: str | None
    candidate_windows: int
    candidate_seconds: float
    close_approaches: int


@dataclass(frozen=True, slots=True)
class ConjunctionEvent:
    """One screened conjunction with its probability.

    Attributes:
        object_id: Secondary identifier.
        approach: The converged close approach solution.
        encounter: Encounter plane geometry at the time of closest approach.
        probability: Probability of collision and its accuracy statement.
    """

    object_id: str
    approach: CloseApproach
    encounter: EncounterGeometry
    probability: ProbabilityResult

    @property
    def tca_s(self) -> float:
        """Time of closest approach, in s from the screening epoch."""
        return self.approach.tca_s

    @property
    def miss_distance_m(self) -> float:
        """Miss distance, in m."""
        return self.approach.miss_distance_m

    @property
    def relative_speed_m_s(self) -> float:
        """Relative speed at the time of closest approach, in m/s."""
        return self.approach.relative_speed_m_s


@dataclass(frozen=True, slots=True)
class ScreeningReport:
    """The structured result of a screening run.

    Attributes:
        threshold_m: Screening threshold applied.
        window_s: Screening window length, in s.
        method: Identifier of the probability method used.
        traces: One entry per secondary, in catalogue order.
        events: Conjunction events, sorted by decreasing probability.
    """

    threshold_m: float
    window_s: float
    method: str
    traces: tuple[PairTrace, ...]
    events: tuple[ConjunctionEvent, ...]

    @property
    def screened(self) -> int:
        """Number of secondaries examined."""
        return len(self.traces)

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Number of pairs rejected by each filter, keyed by filter name."""
        counts: dict[str, int] = {}
        for trace in self.traces:
            if trace.rejected_by is not None:
                counts[trace.rejected_by] = counts.get(trace.rejected_by, 0) + 1
        return counts

    @property
    def survivors(self) -> int:
        """Number of pairs that passed the whole cascade."""
        return sum(1 for trace in self.traces if trace.rejected_by is None)

    @property
    def cascade_cost_windows(self) -> float:
        """Total candidate window duration across all surviving pairs, in s."""
        return float(sum(trace.candidate_seconds for trace in self.traces))


def propagated_position_covariance(
    catalog_object: CatalogObject, epoch_s: float
) -> tuple[Covariance, OrbitState]:
    """Propagate one object's covariance to ``epoch_s`` and return its position block.

    The chain is: rotate the RIC covariance quoted at the object epoch into the
    inertial frame, map it forward with the state transition matrix of the
    two-body flow, then take the leading 3 by 3 position block. Each stage is a
    congruence transform, so symmetry and positive semi-definiteness are preserved
    in exact arithmetic and preserved to rounding in practice.

    Returns:
        The inertial position covariance at ``epoch_s`` and the propagated state.
    """
    inertial_at_epoch = rotate_covariance_to_inertial(
        catalog_object.covariance_ric, catalog_object.state
    )
    transition = state_transition_matrix(
        catalog_object.state, epoch_s - catalog_object.state.epoch_s
    )
    propagated = propagate_covariance(inertial_at_epoch, transition)
    return propagated.position_block(), propagate_to(catalog_object.state, epoch_s)


def build_encounter(
    primary: CatalogObject, secondary: CatalogObject, approach: CloseApproach
) -> EncounterGeometry:
    """Assemble the encounter plane geometry for one close approach.

    The combined relative position covariance is the sum of the two propagated
    position covariances, which assumes the two orbit determination solutions are
    uncorrelated.

    The hard body is handled the same way in both directions. When both objects
    are spheres the combined body is the sphere of the combined radius, its
    shadow on the encounter plane is the disc of that radius whatever the
    approach direction, and no cross section is attached. When either is not, the
    two bodies are combined and the shadow is cast along the relative velocity,
    so the cross section, and with it the probability, depends on the direction
    the secondary comes from.
    """
    primary_covariance, primary_state = propagated_position_covariance(primary, approach.tca_s)
    secondary_covariance, _ = propagated_position_covariance(secondary, approach.tca_s)
    combined = combine_covariances(primary_covariance, secondary_covariance)

    basis = encounter_plane_basis(approach.relative_position_m, approach.relative_velocity_m_s)
    plane_covariance = combined.transformed(basis, frame="encounter")
    miss_vector = project_to_encounter_plane(approach.relative_position_m, basis)

    body = combine_hard_bodies(primary.hard_body, secondary.hard_body)
    section = None if body.is_sphere else projected_cross_section(body, basis)
    radius = (
        primary.radius_m + secondary.radius_m if section is None else section.equivalent_radius_m
    )

    return EncounterGeometry(
        tca_s=approach.tca_s,
        relative_position_m=approach.relative_position_m,
        relative_velocity_m_s=approach.relative_velocity_m_s,
        relative_covariance=combined,
        relative_covariance_ric=rotate_covariance_to_ric(combined, primary_state),
        basis=basis,
        miss_vector_m=miss_vector,
        plane_covariance=plane_covariance,
        hard_body_radius_m=radius,
        cross_section=section,
    )


def run_screening(
    catalog: SyntheticCatalog,
    config: ScreeningConfig | None = None,
    method: ProbabilityMethod | None = None,
    require_converged: bool = True,
) -> ScreeningReport:
    """Screen every secondary in ``catalog`` against its primary.

    Args:
        catalog: Primary and secondaries to screen.
        config: Filter and search tuning.
        method: Probability method, Foster by default.
        require_converged: When true, close approaches whose refinement did not
            converge are dropped rather than reported. A non-converged solution
            is not reproducible across platforms and must not reach a report that
            a regression test pins.

    Returns:
        The screening report, with events sorted by decreasing probability.
    """
    settings = config or ScreeningConfig()
    probability_method: ProbabilityMethod = method or FosterMethod()

    traces: list[PairTrace] = []
    events: list[ConjunctionEvent] = []

    for secondary in catalog.secondaries:
        cascade: CascadeResult = run_cascade(
            catalog.primary.state, secondary.state, settings.cascade
        )
        if not cascade.passed:
            traces.append(
                PairTrace(
                    object_id=secondary.object_id,
                    verdicts=cascade.verdicts,
                    rejected_by=cascade.rejected_by,
                    candidate_windows=0,
                    candidate_seconds=0.0,
                    close_approaches=0,
                )
            )
            continue

        approaches = find_close_approaches(
            catalog.primary.state,
            secondary.state,
            cascade.candidate_windows,
            settings.close_approach,
        )
        usable = [item for item in approaches if item.converged or not require_converged]
        for approach in usable:
            encounter = build_encounter(catalog.primary, secondary, approach)
            if not is_symmetric_positive_semidefinite(encounter.plane_covariance):
                continue
            events.append(
                ConjunctionEvent(
                    object_id=secondary.object_id,
                    approach=approach,
                    encounter=encounter,
                    probability=probability_method.probability(encounter),
                )
            )
        traces.append(
            PairTrace(
                object_id=secondary.object_id,
                verdicts=cascade.verdicts,
                rejected_by=None,
                candidate_windows=len(cascade.candidate_windows),
                candidate_seconds=float(
                    sum(end - begin for begin, end in cascade.candidate_windows)
                ),
                close_approaches=len(usable),
            )
        )

    events.sort(key=lambda item: (-item.probability.value, item.miss_distance_m, item.object_id))
    return ScreeningReport(
        threshold_m=settings.cascade.threshold_m,
        window_s=settings.cascade.window_s,
        method=probability_method.name,
        traces=tuple(traces),
        events=tuple(events),
    )


def report_summary_matrix(report: ScreeningReport) -> np.ndarray:
    """Return one row per event with time, miss distance, relative speed, and probability."""
    rows = [
        [
            event.tca_s,
            event.miss_distance_m,
            event.relative_speed_m_s,
            event.probability.value,
        ]
        for event in report.events
    ]
    return np.array(rows, dtype=np.float64).reshape(-1, 4)
