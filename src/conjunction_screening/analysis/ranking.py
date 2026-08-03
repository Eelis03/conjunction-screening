"""Ranking screened conjunctions and mapping probability onto an action.

The action thresholds follow the two-level convention used by operational
conjunction assessment: a probability at or above 1e-4 warrants planning a
manoeuvre, a probability at or above 1e-7 warrants continued tracking, and
anything below is dismissed. The thresholds are configurable because they are a
policy choice about acceptable risk, not a property of the mathematics. The
probability of collision is a decision input, not a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from conjunction_screening.model.encounter import principal_axis_form
from conjunction_screening.pipeline.screening import ConjunctionEvent, ScreeningReport

__all__ = [
    "ACTION_THRESHOLD",
    "MONITOR_THRESHOLD",
    "ActionClass",
    "ActionThresholds",
    "RankedConjunction",
    "format_covariance_table",
    "format_ranking_table",
    "rank_events",
    "rank_report",
]

ACTION_THRESHOLD: Final[float] = 1e-4
"""Probability at or above which a manoeuvre is planned."""

MONITOR_THRESHOLD: Final[float] = 1e-7
"""Probability at or above which the event stays under observation."""


class ActionClass(StrEnum):
    """What a screened conjunction is escalated to."""

    ACT = "act"
    MONITOR = "monitor"
    DISMISS = "dismiss"


@dataclass(frozen=True, slots=True)
class ActionThresholds:
    """The two probability thresholds separating the three action classes."""

    act: float = ACTION_THRESHOLD
    monitor: float = MONITOR_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 < self.monitor <= self.act < 1.0:
            raise ValueError("thresholds must satisfy 0 < monitor <= act < 1")

    def classify(self, probability: float) -> ActionClass:
        """Map a probability of collision onto an action class."""
        if probability >= self.act:
            return ActionClass.ACT
        if probability >= self.monitor:
            return ActionClass.MONITOR
        return ActionClass.DISMISS


@dataclass(frozen=True, slots=True)
class RankedConjunction:
    """One conjunction in ranked order.

    Attributes:
        rank: One-based position in the ranking.
        object_id: Secondary identifier.
        tca_s: Time of closest approach, in s from the screening epoch.
        miss_distance_m: Miss distance, in m.
        relative_speed_m_s: Relative speed at the time of closest approach.
        hard_body_radius_m: Combined hard body radius, in m.
        sigma_x_m: Larger principal in-plane standard deviation, in m.
        sigma_y_m: Smaller principal in-plane standard deviation, in m.
        normalised_miss_distance: Miss distance in standard deviations of the
            combined covariance. This is what the probability is a function of,
            and it need not order events the same way metres do.
        probability: Probability of collision.
        action: Action class implied by the thresholds.
        converged: Whether both the close approach solve and the probability
            evaluation met their tolerances.
    """

    rank: int
    object_id: str
    tca_s: float
    miss_distance_m: float
    relative_speed_m_s: float
    hard_body_radius_m: float
    sigma_x_m: float
    sigma_y_m: float
    normalised_miss_distance: float
    probability: float
    action: ActionClass
    converged: bool


def rank_events(
    events: tuple[ConjunctionEvent, ...], thresholds: ActionThresholds | None = None
) -> tuple[RankedConjunction, ...]:
    """Sort events by decreasing probability and assign an action to each."""
    limits = thresholds or ActionThresholds()
    ordered = sorted(
        events, key=lambda item: (-item.probability.value, item.miss_distance_m, item.object_id)
    )
    forms = [principal_axis_form(event.encounter) for event in ordered]
    return tuple(
        RankedConjunction(
            rank=index + 1,
            object_id=event.object_id,
            tca_s=event.tca_s,
            miss_distance_m=event.miss_distance_m,
            relative_speed_m_s=event.relative_speed_m_s,
            hard_body_radius_m=event.encounter.hard_body_radius_m,
            sigma_x_m=form.sigma_x_m,
            sigma_y_m=form.sigma_y_m,
            normalised_miss_distance=form.normalised_miss_distance,
            probability=event.probability.value,
            action=limits.classify(event.probability.value),
            converged=event.approach.converged and event.probability.converged,
        )
        for index, (event, form) in enumerate(zip(ordered, forms, strict=True))
    )


def rank_report(
    report: ScreeningReport, thresholds: ActionThresholds | None = None
) -> tuple[RankedConjunction, ...]:
    """Rank the events of a screening report."""
    return rank_events(report.events, thresholds)


def format_ranking_table(ranked: tuple[RankedConjunction, ...], limit: int | None = None) -> str:
    """Render a ranking as a fixed-width text table."""
    header = (
        f"{'rank':>4}  {'object':<14}  {'tca [s]':>12}  {'miss [m]':>10}  "
        f"{'v_rel [m/s]':>11}  {'radius [m]':>10}  {'Pc':>11}  {'action':<8}"
    )
    lines = [header, "-" * len(header)]
    rows = ranked if limit is None else ranked[:limit]
    for item in rows:
        lines.append(
            f"{item.rank:>4}  {item.object_id:<14}  {item.tca_s:>12.3f}  "
            f"{item.miss_distance_m:>10.1f}  {item.relative_speed_m_s:>11.1f}  "
            f"{item.hard_body_radius_m:>10.1f}  {item.probability:>11.4e}  {item.action.value:<8}"
        )
    if limit is not None and len(ranked) > limit:
        lines.append(f"... {len(ranked) - limit} further event(s) not shown")
    return "\n".join(lines)


def format_covariance_table(ranked: tuple[RankedConjunction, ...]) -> str:
    """Render the covariance geometry behind a ranking as a fixed-width text table.

    The ranking table reports the miss distance in metres, which is not the
    quantity the probability depends on. This table reports the same miss
    distance in standard deviations of the combined covariance alongside the two
    principal in-plane sigmas, which is what makes an ordering that looks wrong
    in metres readable.
    """
    header = (
        f"{'object':<14}  {'miss [m]':>10}  {'sigma_x [m]':>11}  {'sigma_y [m]':>11}  "
        f"{'miss / sigma':>12}  {'Pc':>11}"
    )
    lines = [header, "-" * len(header)]
    for item in ranked:
        lines.append(
            f"{item.object_id:<14}  {item.miss_distance_m:>10.1f}  {item.sigma_x_m:>11.1f}  "
            f"{item.sigma_y_m:>11.1f}  {item.normalised_miss_distance:>12.2f}  "
            f"{item.probability:>11.4e}"
        )
    return "\n".join(lines)
