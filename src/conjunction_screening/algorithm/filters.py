"""The conjunction filter cascade.

Three filters are applied in the order published by Hoots, Crawford, and Roehrich
(1984), from cheapest to most expensive:

1. the perigee and apogee filter, which compares the radial shells the two orbits
   occupy;
2. the orbit path filter, which bounds the minimum distance between the two orbit
   paths treated as static curves;
3. the time filter, which asks whether the two objects are ever simultaneously
   inside the parts of their paths that can produce a close approach.

The cascade exists to avoid the cost of a full close approach search on pairs
that provably cannot conjunct. That makes one property non-negotiable: a filter
may reject a pair only when it has proved that no separation below the screening
threshold is possible anywhere in the window. Every rejection in this module is
therefore backed by an inequality that holds for all true anomalies and all
times, never by a sampled minimum. Where sampling is used, the sample is paired
with a Lipschitz bound that converts it into a rigorous bound.

The propagation model is two-body Keplerian, so the orbital elements other than
the anomaly are constant over the window and the static-path argument of filters
2 and 3 is exact. docs/design-notes.md records why a J2 secular model was not
used and what it would cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from conjunction_screening.model.constants import MU_EARTH
from conjunction_screening.model.state import (
    KeplerianElements,
    OrbitState,
    eccentric_to_mean_anomaly,
    elements_from_state,
    path_positions,
    true_to_eccentric_anomaly,
)

__all__ = [
    "PATH_FILTER",
    "PERIGEE_APOGEE_FILTER",
    "TIME_FILTER",
    "CascadeResult",
    "CascadeSettings",
    "FilterVerdict",
    "PathSeparation",
    "TimeInterval",
    "at_risk_arcs",
    "minimum_path_separation",
    "orbit_path_filter",
    "perigee_apogee_filter",
    "run_cascade",
    "time_filter",
]

PERIGEE_APOGEE_FILTER: Final[str] = "perigee-apogee"
PATH_FILTER: Final[str] = "orbit-path"
TIME_FILTER: Final[str] = "time"

_TWO_PI: Final[float] = 2.0 * np.pi

TimeInterval = tuple[float, float]
"""A closed time interval in seconds from the screening epoch."""


@dataclass(frozen=True, slots=True)
class CascadeSettings:
    """Tuning for the filter cascade.

    Attributes:
        threshold_m: Screening threshold. A pair may be rejected only if no
            separation at or below this value is achievable.
        window_s: Length of the screening window from the epoch, in s.
        path_divisions: Divisions per axis of the initial branch and bound grid
            over the two true anomalies.
        path_max_cells: Cell budget for the branch and bound. Exceeding it makes
            the filter pass the pair, which is the conservative direction.
        path_pad_fraction: The branch and bound stops refining once the Lipschitz
            pad falls below this fraction of the threshold. A surviving cell at
            that point makes the filter pass the pair, so the filter behaves as
            if the threshold were larger by this fraction.
        arc_samples: Samples per orbit used to locate the at-risk true anomaly
            arcs consumed by the time filter.
        max_intervals: Cap on the number of time intervals per object. Exceeding
            it makes the time filter pass the pair.
    """

    threshold_m: float = 5_000.0
    window_s: float = 86_400.0
    path_divisions: int = 128
    path_max_cells: int = 200_000
    path_pad_fraction: float = 0.05
    arc_samples: int = 512
    max_intervals: int = 4_000

    def __post_init__(self) -> None:
        if not self.threshold_m > 0.0:
            raise ValueError("threshold_m must be positive")
        if not self.window_s > 0.0:
            raise ValueError("window_s must be positive")
        if self.path_divisions < 4:
            raise ValueError("path_divisions must be at least 4")
        if self.arc_samples < 16:
            raise ValueError("arc_samples must be at least 16")


@dataclass(frozen=True, slots=True)
class FilterVerdict:
    """The outcome of one filter on one pair.

    Attributes:
        name: Filter identifier.
        passed: True when the filter could not rule the pair out.
        bound: The conservative quantity the filter computed. For the distance
            filters it is a lower bound on the achievable separation in m; for
            the time filter it is the smallest gap in s between the two objects'
            at-risk intervals, and zero when they overlap.
        threshold: The value ``bound`` was compared against, in the same units.
        units: ``"m"`` or ``"s"``.
        detail: Human-readable justification recorded in the screening trace.
    """

    name: str
    passed: bool
    bound: float
    threshold: float
    units: str
    detail: str


@dataclass(frozen=True, slots=True)
class PathSeparation:
    """Result of the branch and bound search over the two orbit paths.

    Attributes:
        lower_bound_m: A value the true minimum path separation cannot fall below.
        witness_m: The smallest sampled separation seen, an upper bound on the
            true minimum.
        resolved_pad_m: The Lipschitz pad at the level where the search stopped.
        cells_evaluated: Total cells evaluated, reported for cost accounting.
        budget_exhausted: True when the search stopped on the cell or pad budget
            rather than on a decision.
        can_approach: True when a separation at or below the threshold could not
            be ruled out.
    """

    lower_bound_m: float
    witness_m: float
    resolved_pad_m: float
    cells_evaluated: int
    budget_exhausted: bool
    can_approach: bool


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """The verdicts of the whole cascade on one pair.

    Attributes:
        verdicts: Verdicts in application order, truncated at the first rejection.
        candidate_windows: Time intervals in s from the epoch during which a close
            approach is possible. Empty when the pair was rejected.
    """

    verdicts: tuple[FilterVerdict, ...]
    candidate_windows: tuple[TimeInterval, ...]

    @property
    def passed(self) -> bool:
        """True when every filter that ran passed the pair."""
        return all(verdict.passed for verdict in self.verdicts)

    @property
    def rejected_by(self) -> str | None:
        """Name of the filter that rejected the pair, or None if none did."""
        for verdict in self.verdicts:
            if not verdict.passed:
                return verdict.name
        return None


def perigee_apogee_filter(
    primary: KeplerianElements, secondary: KeplerianElements, threshold_m: float
) -> FilterVerdict:
    """Reject pairs whose radial shells are separated by more than the threshold.

    Object one is confined to the shell ``[q1, Q1]`` and object two to ``[q2, Q2]``
    for all time. For any pair of positions the separation obeys the reverse
    triangle inequality, ``|r1 - r2| >= | |r1| - |r2| |``, so if the shells are
    disjoint the separation is at least the gap between them. The gap is
    ``max(q2 - Q1, q1 - Q2)`` and is negative when the shells overlap. Rejecting
    when the gap exceeds the threshold is therefore exact, not approximate.
    """
    gap = max(
        secondary.perigee_radius_m - primary.apogee_radius_m,
        primary.perigee_radius_m - secondary.apogee_radius_m,
    )
    passed = gap <= threshold_m
    detail = (
        f"radial shells [{primary.perigee_radius_m / 1e3:.1f}, "
        f"{primary.apogee_radius_m / 1e3:.1f}] km and "
        f"[{secondary.perigee_radius_m / 1e3:.1f}, {secondary.apogee_radius_m / 1e3:.1f}] km "
        f"leave a gap of {gap / 1e3:.1f} km"
    )
    return FilterVerdict(
        name=PERIGEE_APOGEE_FILTER,
        passed=passed,
        bound=gap,
        threshold=threshold_m,
        units="m",
        detail=detail,
    )


def _path_separations(
    primary: KeplerianElements,
    secondary: KeplerianElements,
    primary_anomalies: np.ndarray,
    secondary_anomalies: np.ndarray,
) -> np.ndarray:
    first = path_positions(primary, primary_anomalies)
    second = path_positions(secondary, secondary_anomalies)
    return np.asarray(np.linalg.norm(first - second, axis=-1), dtype=np.float64)


def _subdivide(
    primary_anomalies: np.ndarray, secondary_anomalies: np.ndarray, half_width: float
) -> tuple[np.ndarray, np.ndarray]:
    quarter = 0.5 * half_width
    offsets = np.array(
        [[-quarter, -quarter], [-quarter, quarter], [quarter, -quarter], [quarter, quarter]],
        dtype=np.float64,
    )
    first = (primary_anomalies[:, None] + offsets[None, :, 0]).ravel()
    second = (secondary_anomalies[:, None] + offsets[None, :, 1]).ravel()
    return first, second


def minimum_path_separation(
    primary: KeplerianElements,
    secondary: KeplerianElements,
    threshold_m: float,
    settings: CascadeSettings | None = None,
) -> PathSeparation:
    """Decide whether two orbit paths can approach within ``threshold_m``.

    The separation ``d(nu1, nu2) = |r1(nu1) - r2(nu2)|`` is Lipschitz in each
    argument with constant ``KeplerianElements.path_speed_bound_m_rad``, which is
    a closed-form bound valid for every true anomaly. Over a cell of half width
    ``w`` in both arguments the separation therefore cannot fall below
    ``d(centre) - (L1 + L2) w``. That inequality drives a branch and bound:

    * if any cell centre already sits at or below the threshold, a close approach
      is possible and the search stops with a witness;
    * a cell whose lower bound exceeds the threshold is discarded, and when every
      cell is discarded the pair is proved safe;
    * otherwise the surviving cells are quartered and the pad halves.

    The search stops on a witness, on an empty survivor set, on the pad budget,
    or on the cell budget. The last two return ``can_approach=True``, which keeps
    a budget exhaustion from ever turning into a missed conjunction.
    """
    config = settings or CascadeSettings(threshold_m=threshold_m)
    lipschitz = primary.path_speed_bound_m_rad + secondary.path_speed_bound_m_rad
    divisions = config.path_divisions
    spacing = _TWO_PI / divisions
    half_width = 0.5 * spacing
    centres = (np.arange(divisions, dtype=np.float64) + 0.5) * spacing
    first = np.repeat(centres, divisions)
    second = np.tile(centres, divisions)

    pad_floor = max(config.path_pad_fraction * threshold_m, 1.0)
    evaluated = 0
    while True:
        separations = _path_separations(primary, secondary, first, second)
        evaluated += int(separations.size)
        witness = float(np.min(separations))
        pad = lipschitz * half_width
        lower_bound = witness - pad

        if witness <= threshold_m:
            return PathSeparation(
                lower_bound_m=max(lower_bound, 0.0),
                witness_m=witness,
                resolved_pad_m=pad,
                cells_evaluated=evaluated,
                budget_exhausted=False,
                can_approach=True,
            )

        survivors = separations - pad <= threshold_m
        surviving = int(np.count_nonzero(survivors))
        if surviving == 0:
            return PathSeparation(
                lower_bound_m=lower_bound,
                witness_m=witness,
                resolved_pad_m=pad,
                cells_evaluated=evaluated,
                budget_exhausted=False,
                can_approach=False,
            )
        if pad <= pad_floor or surviving * 4 > config.path_max_cells:
            return PathSeparation(
                lower_bound_m=lower_bound,
                witness_m=witness,
                resolved_pad_m=pad,
                cells_evaluated=evaluated,
                budget_exhausted=True,
                can_approach=True,
            )

        first, second = _subdivide(first[survivors], second[survivors], half_width)
        half_width *= 0.5


def orbit_path_filter(
    primary: KeplerianElements,
    secondary: KeplerianElements,
    threshold_m: float,
    settings: CascadeSettings | None = None,
) -> FilterVerdict:
    """Reject pairs whose orbit paths never come within the screening threshold."""
    separation = minimum_path_separation(primary, secondary, threshold_m, settings)
    if separation.can_approach and not separation.budget_exhausted:
        detail = (
            f"paths approach to {separation.witness_m / 1e3:.3f} km, "
            f"inside the {threshold_m / 1e3:.3f} km threshold"
        )
    elif separation.budget_exhausted:
        detail = (
            f"search budget reached with pad {separation.resolved_pad_m:.1f} m; "
            f"separation not ruled out below {threshold_m / 1e3:.3f} km"
        )
    else:
        detail = (
            f"paths are at least {separation.lower_bound_m / 1e3:.3f} km apart, "
            f"outside the {threshold_m / 1e3:.3f} km threshold"
        )
    return FilterVerdict(
        name=PATH_FILTER,
        passed=separation.can_approach,
        bound=separation.lower_bound_m,
        threshold=threshold_m,
        units="m",
        detail=detail,
    )


def _circular_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start_index, length)`` for each maximal run of True, wrapping at the end."""
    count = int(mask.size)
    if not bool(np.any(mask)):
        return []
    if bool(np.all(mask)):
        return [(0, count)]
    previous = np.roll(mask, 1)
    runs: list[tuple[int, int]] = []
    for start in np.flatnonzero(mask & ~previous):
        length = 1
        while bool(mask[(int(start) + length) % count]):
            length += 1
        runs.append((int(start), length))
    return runs


def at_risk_arcs(
    primary: KeplerianElements,
    secondary: KeplerianElements,
    threshold_m: float,
    settings: CascadeSettings | None = None,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Return the true anomaly arcs on each orbit where a close approach is possible.

    Both paths are sampled uniformly in true anomaly. A sample on orbit one is
    marked when its distance to the nearest sample on orbit two is at or below
    ``threshold_m`` plus a Lipschitz pad of ``L1 dnu1 / 2 + L2 dnu2 / 2``. That
    pad is exactly the worst-case error introduced by sampling, so any true pair
    ``(nu1*, nu2*)`` with separation at or below the threshold has its nearest
    samples marked. Marked runs are then widened by one full sample spacing on
    each side, which more than covers the half spacing between a true anomaly and
    its nearest sample.

    Returns:
        Arcs on the primary and arcs on the secondary, each as ``(start, end)``
        pairs of true anomaly in rad with ``end >= start`` and ``end - start`` at
        most ``2 pi``.
    """
    config = settings or CascadeSettings(threshold_m=threshold_m)
    samples = config.arc_samples
    spacing = _TWO_PI / samples
    centres = (np.arange(samples, dtype=np.float64) + 0.5) * spacing
    pad = 0.5 * spacing * (primary.path_speed_bound_m_rad + secondary.path_speed_bound_m_rad)
    limit = threshold_m + pad

    first_positions = path_positions(primary, centres)
    second_positions = path_positions(secondary, centres)
    primary_mask = np.zeros(samples, dtype=bool)
    secondary_mask = np.zeros(samples, dtype=bool)
    block = 64
    for start in range(0, samples, block):
        stop = min(start + block, samples)
        distances = np.linalg.norm(
            first_positions[start:stop, None, :] - second_positions[None, :, :], axis=-1
        )
        close = distances <= limit
        primary_mask[start:stop] = np.any(close, axis=1)
        secondary_mask |= np.any(close, axis=0)

    return (
        _arcs_from_mask(primary_mask, centres, spacing),
        _arcs_from_mask(secondary_mask, centres, spacing),
    )


def _arcs_from_mask(
    mask: np.ndarray, centres: np.ndarray, spacing: float
) -> tuple[tuple[float, float], ...]:
    arcs: list[tuple[float, float]] = []
    for start, length in _circular_runs(mask):
        begin = float(centres[start]) - spacing
        extent = (length - 1) * spacing + 2.0 * spacing
        arcs.append((begin, begin + min(extent, _TWO_PI)))
    return tuple(arcs)


def _mean_anomaly_at_epoch(elements: KeplerianElements) -> float:
    ecc = elements.eccentricity
    return float(
        eccentric_to_mean_anomaly(true_to_eccentric_anomaly(elements.true_anomaly_rad, ecc), ecc)
    )


def _arc_intervals(
    elements: KeplerianElements, arc: tuple[float, float], window_s: float
) -> list[TimeInterval]:
    """Convert one true anomaly arc into the time intervals it occupies in the window."""
    ecc = elements.eccentricity
    mean_motion = elements.mean_motion_rad_s
    period = elements.period_s
    start_mean = float(eccentric_to_mean_anomaly(true_to_eccentric_anomaly(arc[0], ecc), ecc))
    span_nu = arc[1] - arc[0]
    if span_nu >= _TWO_PI:
        return [(0.0, window_s)]
    end_mean = float(eccentric_to_mean_anomaly(true_to_eccentric_anomaly(arc[1], ecc), ecc))
    span_mean = (end_mean - start_mean) % _TWO_PI
    duration = span_mean / mean_motion

    offset = ((start_mean - _mean_anomaly_at_epoch(elements)) / mean_motion) % period
    intervals: list[TimeInterval] = []
    first_index = int(np.floor((0.0 - offset - duration) / period)) - 1
    last_index = int(np.ceil((window_s - offset) / period)) + 1
    for index in range(first_index, last_index + 1):
        begin = offset + index * period
        end = begin + duration
        if end < 0.0 or begin > window_s:
            continue
        intervals.append((max(begin, 0.0), min(end, window_s)))
    return intervals


def _merge_intervals(intervals: list[TimeInterval]) -> list[TimeInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for begin, end in ordered[1:]:
        last_begin, last_end = merged[-1]
        if begin <= last_end:
            merged[-1] = (last_begin, max(last_end, end))
        else:
            merged.append((begin, end))
    return merged


def _overlap(
    first: list[TimeInterval], second: list[TimeInterval]
) -> tuple[list[TimeInterval], float]:
    """Return the intersection of two disjoint sorted interval lists and their smallest gap.

    Standard two-pointer intersection. When a pair of intervals does not overlap
    the quantity ``low - high`` is exactly the gap between them, so the smallest
    value seen over the sweep is the smallest gap between the two families.
    """
    overlaps: list[TimeInterval] = []
    smallest_gap = float("inf")
    left = 0
    right = 0
    while left < len(first) and right < len(second):
        low = max(first[left][0], second[right][0])
        high = min(first[left][1], second[right][1])
        if low <= high:
            overlaps.append((low, high))
            smallest_gap = 0.0
        else:
            smallest_gap = min(smallest_gap, low - high)
        if first[left][1] < second[right][1]:
            left += 1
        else:
            right += 1
    return _merge_intervals(overlaps), smallest_gap


def time_filter(
    primary: KeplerianElements,
    secondary: KeplerianElements,
    threshold_m: float,
    settings: CascadeSettings | None = None,
) -> tuple[FilterVerdict, tuple[TimeInterval, ...]]:
    """Reject pairs that are never simultaneously inside their at-risk arcs.

    A close approach at time ``t`` requires the primary to be somewhere on the
    part of its path that comes within the threshold of the secondary's path, and
    the secondary to be on the matching part of its own path, at the same instant.
    Each arc maps to a periodic family of time intervals through Kepler's
    equation. If no interval of the primary intersects any interval of the
    secondary inside the window, no close approach can occur.

    The construction treats the primary arcs and the secondary arcs as
    independent sets rather than tracking which primary arc pairs with which
    secondary arc. That over-approximates the at-risk set, which is the safe
    direction.

    Returns:
        The verdict and the candidate time windows, in s from the epoch.
    """
    config = settings or CascadeSettings(threshold_m=threshold_m, window_s=86_400.0)
    primary_arcs, secondary_arcs = at_risk_arcs(primary, secondary, threshold_m, config)
    if not primary_arcs or not secondary_arcs:
        return (
            FilterVerdict(
                name=TIME_FILTER,
                passed=False,
                bound=float("inf"),
                threshold=0.0,
                units="s",
                detail="no at-risk arc exists on at least one of the two orbits",
            ),
            (),
        )

    primary_intervals: list[TimeInterval] = []
    for arc in primary_arcs:
        primary_intervals.extend(_arc_intervals(primary, arc, config.window_s))
    secondary_intervals: list[TimeInterval] = []
    for arc in secondary_arcs:
        secondary_intervals.extend(_arc_intervals(secondary, arc, config.window_s))

    if len(primary_intervals) > config.max_intervals or len(secondary_intervals) > (
        config.max_intervals
    ):
        whole = ((0.0, config.window_s),)
        return (
            FilterVerdict(
                name=TIME_FILTER,
                passed=True,
                bound=0.0,
                threshold=0.0,
                units="s",
                detail="interval budget reached; the whole window is treated as a candidate",
            ),
            whole,
        )

    merged_primary = _merge_intervals(primary_intervals)
    merged_secondary = _merge_intervals(secondary_intervals)
    windows, gap = _overlap(merged_primary, merged_secondary)
    passed = bool(windows)
    if passed:
        covered = sum(end - begin for begin, end in windows)
        detail = (
            f"{len(windows)} overlapping window(s) covering {covered:.1f} s "
            f"of the {config.window_s:.0f} s screening window"
        )
    else:
        detail = (
            f"at-risk arcs never coincide; the closest the two objects come to "
            f"being simultaneously at risk is {gap:.1f} s"
        )
    return (
        FilterVerdict(
            name=TIME_FILTER,
            passed=passed,
            bound=0.0 if passed else gap,
            threshold=0.0,
            units="s",
            detail=detail,
        ),
        tuple(windows),
    )


def run_cascade(
    primary: OrbitState,
    secondary: OrbitState,
    settings: CascadeSettings | None = None,
    gravitational_parameter: float = MU_EARTH,
) -> CascadeResult:
    """Apply the three filters in order, stopping at the first rejection."""
    config = settings or CascadeSettings()
    primary_elements = elements_from_state(primary, gravitational_parameter)
    secondary_elements = elements_from_state(secondary, gravitational_parameter)

    verdicts: list[FilterVerdict] = []
    first = perigee_apogee_filter(primary_elements, secondary_elements, config.threshold_m)
    verdicts.append(first)
    if not first.passed:
        return CascadeResult(verdicts=tuple(verdicts), candidate_windows=())

    second = orbit_path_filter(
        primary_elements, secondary_elements, config.threshold_m, config
    )
    verdicts.append(second)
    if not second.passed:
        return CascadeResult(verdicts=tuple(verdicts), candidate_windows=())

    third, windows = time_filter(
        primary_elements, secondary_elements, config.threshold_m, config
    )
    verdicts.append(third)
    return CascadeResult(verdicts=tuple(verdicts), candidate_windows=windows)
