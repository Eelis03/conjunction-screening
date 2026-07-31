"""Time of closest approach and miss distance.

A close approach is a local minimum of the relative range. Range is awkward to
differentiate near zero, so the search works on

    g(t) = dr . dv

which is the range rate multiplied by the range and therefore vanishes at every
range extremum while staying smooth through them. A coarse sweep brackets each
sign change of ``g`` from negative to positive, which is a minimum rather than a
maximum, and Brent's method refines the root inside that bracket.

The coarse step must be small enough that no minimum falls entirely between two
samples. The step is therefore derived from the geometry rather than chosen: for
a relative speed ``v`` the range changes by at most ``v h`` over one step, so a
step of ``threshold / v`` guarantees that any approach reaching the threshold is
visible at a sample. The default configuration applies that rule with a safety
factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import brentq

from conjunction_screening.algorithm.propagation import propagate_many
from conjunction_screening.model.arrays import Vector
from conjunction_screening.model.constants import MU_EARTH
from conjunction_screening.model.state import OrbitState

__all__ = [
    "CloseApproach",
    "CloseApproachSettings",
    "coarse_step_for",
    "find_close_approaches",
    "relative_state",
]

_MIN_COARSE_STEP_S: Final[float] = 1e-3
"""Floor on the coarse step so a degenerate geometry cannot produce a zero step."""


@dataclass(frozen=True, slots=True)
class CloseApproachSettings:
    """Tuning for the close approach search.

    Attributes:
        threshold_m: Approaches wider than this are discarded.
        step_safety: Fraction of ``threshold_m / relative_speed`` used as the
            coarse sweep step. A value below one means several samples fall
            inside every approach that reaches the threshold.
        max_step_s: Upper limit on the coarse step, applied so that a slow
            encounter still gets a reasonable number of samples.
        time_tolerance_s: Absolute tolerance passed to Brent's method. The
            residual range rate at the returned time is bounded by roughly
            ``(v^2 / d) * time_tolerance_s``, which is how the test suite derives
            its tolerance on that residual.
        max_iterations: Iteration cap for Brent's method.
    """

    threshold_m: float = 5_000.0
    step_safety: float = 0.25
    max_step_s: float = 10.0
    time_tolerance_s: float = 1e-9
    max_iterations: int = 200

    def __post_init__(self) -> None:
        if not 0.0 < self.step_safety <= 1.0:
            raise ValueError("step_safety must lie in (0, 1]")
        if not self.time_tolerance_s > 0.0:
            raise ValueError("time_tolerance_s must be positive")


@dataclass(frozen=True, slots=True)
class CloseApproach:
    """One converged or attempted close approach solution.

    Attributes:
        tca_s: Time of closest approach in s from the screening epoch.
        miss_distance_m: Range between the two objects at that time, in m.
        relative_speed_m_s: Relative speed at that time, in m/s.
        relative_position_m: Secondary minus primary position at that time.
        relative_velocity_m_s: Secondary minus primary velocity at that time.
        range_rate_m_s: Residual range rate at the returned time. Zero to within
            the refinement tolerance for a converged solution.
        converged: Whether Brent's method reported convergence. Results with
            this flag false must not be pinned in a regression test, because a
            non-converged iterate depends on floating point reduction order and
            differs between platforms.
        bracket_s: The coarse interval the root was found in.
    """

    tca_s: float
    miss_distance_m: float
    relative_speed_m_s: float
    relative_position_m: Vector
    relative_velocity_m_s: Vector
    range_rate_m_s: float
    converged: bool
    bracket_s: tuple[float, float]


def relative_state(
    primary: OrbitState,
    secondary: OrbitState,
    times_s: np.ndarray,
    gravitational_parameter: float = MU_EARTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative positions and velocities at absolute times ``times_s``.

    Both states are propagated from their own epochs, so the two need not share
    an epoch.
    """
    absolute = np.asarray(times_s, dtype=np.float64)
    primary_position, primary_velocity = propagate_many(
        primary, absolute - primary.epoch_s, gravitational_parameter
    )
    secondary_position, secondary_velocity = propagate_many(
        secondary, absolute - secondary.epoch_s, gravitational_parameter
    )
    return secondary_position - primary_position, secondary_velocity - primary_velocity


def coarse_step_for(
    primary: OrbitState, secondary: OrbitState, settings: CloseApproachSettings
) -> float:
    """Return the coarse sweep step in s implied by the geometry and the settings.

    The relative speed is bounded above by the sum of the two speeds at perigee,
    which for a two-body orbit is the largest speed either object attains. Using
    that bound rather than the instantaneous relative speed keeps the step valid
    over the whole window.
    """
    speed_bound = primary.speed_m_s + secondary.speed_m_s
    if speed_bound <= 0.0:
        return settings.max_step_s
    step = settings.step_safety * settings.threshold_m / speed_bound
    return float(min(max(step, _MIN_COARSE_STEP_S), settings.max_step_s))


def find_close_approaches(
    primary: OrbitState,
    secondary: OrbitState,
    windows: tuple[tuple[float, float], ...],
    settings: CloseApproachSettings | None = None,
    gravitational_parameter: float = MU_EARTH,
) -> tuple[CloseApproach, ...]:
    """Find every close approach inside ``windows`` that reaches the threshold.

    Args:
        primary: Primary object state.
        secondary: Secondary object state.
        windows: Candidate time intervals in s from the screening epoch, as
            produced by the filter cascade.
        settings: Search tuning.
        gravitational_parameter: Central body mu in m^3 / s^2.

    Returns:
        Close approaches sorted by time, each with a miss distance at or below
        the threshold.
    """
    config = settings or CloseApproachSettings()
    step = coarse_step_for(primary, secondary, config)
    found: list[CloseApproach] = []

    for begin, end in windows:
        if end <= begin:
            continue
        count = max(int(np.ceil((end - begin) / step)) + 1, 3)
        samples = np.linspace(begin, end, count)
        offsets, rates = relative_state(primary, secondary, samples, gravitational_parameter)
        projection = np.einsum("ij,ij->i", offsets, rates)

        for index in range(count - 1):
            left = float(projection[index])
            right = float(projection[index + 1])
            if left < 0.0 <= right:
                approach = _refine(
                    primary,
                    secondary,
                    float(samples[index]),
                    float(samples[index + 1]),
                    config,
                    gravitational_parameter,
                )
            elif left == 0.0 and index == 0:
                approach = _evaluate(
                    primary,
                    secondary,
                    float(samples[index]),
                    (float(samples[index]), float(samples[index])),
                    True,
                    gravitational_parameter,
                )
            else:
                continue
            if approach.miss_distance_m <= config.threshold_m:
                found.append(approach)

    found.sort(key=lambda item: item.tca_s)
    return tuple(_deduplicate(found, step))


def _projection(
    primary: OrbitState,
    secondary: OrbitState,
    time_s: float,
    gravitational_parameter: float,
) -> float:
    offsets, rates = relative_state(
        primary, secondary, np.array([time_s], dtype=np.float64), gravitational_parameter
    )
    return float(np.dot(offsets[0], rates[0]))


def _refine(
    primary: OrbitState,
    secondary: OrbitState,
    left_s: float,
    right_s: float,
    settings: CloseApproachSettings,
    gravitational_parameter: float,
) -> CloseApproach:
    def objective(time_s: float) -> float:
        return _projection(primary, secondary, time_s, gravitational_parameter)

    root, result = brentq(
        objective,
        left_s,
        right_s,
        xtol=settings.time_tolerance_s,
        maxiter=settings.max_iterations,
        full_output=True,
        disp=False,
    )
    return _evaluate(
        primary,
        secondary,
        float(root),
        (left_s, right_s),
        bool(result.converged),
        gravitational_parameter,
    )


def _evaluate(
    primary: OrbitState,
    secondary: OrbitState,
    time_s: float,
    bracket_s: tuple[float, float],
    converged: bool,
    gravitational_parameter: float,
) -> CloseApproach:
    offsets, rates = relative_state(
        primary, secondary, np.array([time_s], dtype=np.float64), gravitational_parameter
    )
    offset = offsets[0]
    rate = rates[0]
    distance = float(np.linalg.norm(offset))
    range_rate = float(np.dot(offset, rate)) / distance if distance > 0.0 else 0.0
    return CloseApproach(
        tca_s=time_s,
        miss_distance_m=distance,
        relative_speed_m_s=float(np.linalg.norm(rate)),
        relative_position_m=offset,
        relative_velocity_m_s=rate,
        range_rate_m_s=range_rate,
        converged=converged,
        bracket_s=bracket_s,
    )


def _deduplicate(approaches: list[CloseApproach], step_s: float) -> list[CloseApproach]:
    """Drop approaches that repeat because candidate windows overlapped."""
    unique: list[CloseApproach] = []
    for approach in approaches:
        if unique and abs(approach.tca_s - unique[-1].tca_s) <= step_s:
            if approach.miss_distance_m < unique[-1].miss_distance_m:
                unique[-1] = approach
            continue
        unique.append(approach)
    return unique
