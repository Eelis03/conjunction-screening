"""Synthetic catalogue generation.

No orbital element set is downloaded and none is embedded. The catalogue is
generated from a seeded random number generator, so a run is reproducible from
its seed alone and the repository carries no third-party data.

Two populations are generated. Background objects are drawn from plausible low
Earth orbit element distributions and mostly never come near the primary. Planted
objects are constructed backwards from a chosen encounter: a time, a miss
distance, and a relative velocity are picked first, the secondary state at that
time is written down, and the two-body propagator is run backwards to the epoch.
Because two-body motion is time reversible, the resulting catalogue object
reproduces the chosen encounter exactly. Those known answers are what the filter
conservativeness tests are asserted against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from conjunction_screening.algorithm.propagation import propagate
from conjunction_screening.model.arrays import Matrix, Vector, unit_vector
from conjunction_screening.model.constants import EARTH_RADIUS_M, MU_EARTH
from conjunction_screening.model.covariance import (
    Covariance,
    ElementSigmas,
    covariance_from_element_sigmas,
)
from conjunction_screening.model.frames import rotate_covariance_to_ric
from conjunction_screening.model.hardbody import HardBody
from conjunction_screening.model.state import (
    KeplerianElements,
    OrbitState,
    elements_from_state,
    state_from_elements,
)

__all__ = [
    "CatalogObject",
    "PlantedConjunction",
    "SyntheticCatalog",
    "catalog_element_table",
    "default_primary",
    "generate_catalog",
    "ric_covariance_from_elements",
]

_MIN_PERIGEE_ALTITUDE_M: Final[float] = 250e3
_MAX_APOGEE_ALTITUDE_M: Final[float] = 3_000e3


@dataclass(frozen=True, slots=True)
class CatalogObject:
    """One catalogued object with its state, size, and uncertainty.

    Attributes:
        object_id: Identifier used in reports.
        state: Inertial state at the screening epoch.
        radius_m: Hard body radius, in m. When ``shape`` is set this is the
            radius of the sphere of the same volume, kept because every report
            prints one number for size.
        covariance_ric: 6 by 6 state covariance in the RIC frame of ``state``,
            with position in m^2 and velocity in m^2/s^2, as a conjunction data
            message quotes it.
        shape: Hard body geometry. None means the sphere of ``radius_m``, which
            is what the synthetic catalogue generates and what a screening
            catalogue normally carries.
    """

    object_id: str
    state: OrbitState
    radius_m: float
    covariance_ric: Covariance
    shape: HardBody | None = None

    def __post_init__(self) -> None:
        if not self.radius_m > 0.0:
            raise ValueError("radius_m must be positive")
        if self.covariance_ric.dimension != 6:
            raise ValueError("covariance_ric must be a 6 by 6 state covariance")

    @property
    def elements(self) -> KeplerianElements:
        """Classical elements of this object at the screening epoch."""
        return elements_from_state(self.state)

    @property
    def hard_body(self) -> HardBody:
        """The hard body geometry, defaulting to the sphere of ``radius_m``."""
        return self.shape if self.shape is not None else HardBody.sphere(self.radius_m)


@dataclass(frozen=True, slots=True)
class PlantedConjunction:
    """Ground truth for one deliberately constructed encounter.

    Attributes:
        object_id: Identifier of the secondary that was constructed.
        tca_s: The time of closest approach the object was built around, in s.
        miss_distance_m: The miss distance the object was built around, in m.
        relative_speed_m_s: Relative speed at that time, in m/s.
    """

    object_id: str
    tca_s: float
    miss_distance_m: float
    relative_speed_m_s: float


@dataclass(frozen=True, slots=True)
class SyntheticCatalog:
    """A primary object, a set of secondaries, and the ground truth for the planted ones."""

    primary: CatalogObject
    secondaries: tuple[CatalogObject, ...]
    planted: tuple[PlantedConjunction, ...]
    seed: int
    window_s: float

    @property
    def size(self) -> int:
        """Number of secondary objects."""
        return len(self.secondaries)


def ric_covariance_from_elements(
    elements: KeplerianElements, sigmas: ElementSigmas, state: OrbitState
) -> Covariance:
    """Build a 6 by 6 RIC state covariance from uncorrelated element uncertainties.

    A conjunction data message quotes its covariance in the RIC frame, so that is
    how a catalogue object stores it. It is constructed in element space and
    rotated, for the reason given in
    :func:`conjunction_screening.model.covariance.covariance_from_element_sigmas`.
    """
    return rotate_covariance_to_ric(covariance_from_element_sigmas(elements, sigmas), state)


def default_primary(object_id: str = "PRIMARY") -> CatalogObject:
    """Return the reference primary, a 700 km sun-synchronous-like circular orbit.

    The element uncertainties are those of a well-tracked operational satellite:
    a semi-major axis known to a few metres, angles known to a few hundred metres
    of arc at orbit radius.
    """
    elements = KeplerianElements(
        semi_major_axis_m=EARTH_RADIUS_M + 700e3,
        eccentricity=1.2e-3,
        inclination_rad=np.deg2rad(98.2),
        raan_rad=np.deg2rad(24.0),
        arg_perigee_rad=np.deg2rad(52.0),
        true_anomaly_rad=np.deg2rad(11.0),
        gravitational_parameter=MU_EARTH,
    )
    state = state_from_elements(elements, epoch_s=0.0)
    sigmas = ElementSigmas(
        semi_major_axis_m=6.0,
        eccentricity=8.0e-7,
        inclination_rad=6.0e-6,
        raan_rad=6.0e-6,
        arg_perigee_rad=2.0e-5,
        mean_anomaly_rad=2.0e-5,
    )
    return CatalogObject(
        object_id=object_id,
        state=state,
        radius_m=5.0,
        covariance_ric=ric_covariance_from_elements(elements, sigmas, state),
    )


def _random_unit(generator: np.random.Generator) -> Vector:
    while True:
        candidate = generator.standard_normal(3)
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-6:
            return np.asarray(candidate / norm, dtype=np.float64)


def _rotate_about(vector: Vector, axis: Vector, angle_rad: float) -> Vector:
    """Rotate ``vector`` about the unit vector ``axis`` by ``angle_rad`` (Rodrigues)."""
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.asarray(
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * float(np.dot(axis, vector)) * (1.0 - cosine),
        dtype=np.float64,
    )


def _perpendicular_unit(generator: np.random.Generator, direction: Vector) -> Vector:
    unit_direction = unit_vector(direction, "direction")
    while True:
        candidate = _random_unit(generator)
        residual = candidate - float(np.dot(candidate, unit_direction)) * unit_direction
        norm = float(np.linalg.norm(residual))
        if norm > 1e-3:
            return np.asarray(residual / norm, dtype=np.float64)


def _covariance_for(
    generator: np.random.Generator, elements: KeplerianElements, state: OrbitState
) -> Covariance:
    """Draw the element uncertainties of a tracked debris object and build its covariance.

    The ranges span a well-tracked object at the tight end and a sparsely tracked
    one at the loose end. The angular sigmas correspond to between about 70 m and
    about 1400 m at orbit radius.
    """
    sigmas = ElementSigmas(
        semi_major_axis_m=float(generator.uniform(4.0, 60.0)),
        eccentricity=float(generator.uniform(5.0e-7, 6.0e-6)),
        inclination_rad=float(generator.uniform(1.0e-5, 8.0e-5)),
        raan_rad=float(generator.uniform(1.0e-5, 8.0e-5)),
        arg_perigee_rad=float(generator.uniform(1.0e-5, 1.4e-4)),
        mean_anomaly_rad=float(generator.uniform(1.0e-5, 1.4e-4)),
    )
    return ric_covariance_from_elements(elements, sigmas, state)


def _background_object(generator: np.random.Generator, object_id: str) -> CatalogObject:
    elements = KeplerianElements(
        semi_major_axis_m=EARTH_RADIUS_M + float(generator.uniform(400e3, 1_400e3)),
        eccentricity=float(generator.uniform(1.0e-4, 0.015)),
        inclination_rad=float(generator.uniform(0.35, np.pi - 0.35)),
        raan_rad=float(generator.uniform(0.0, 2.0 * np.pi)),
        arg_perigee_rad=float(generator.uniform(0.0, 2.0 * np.pi)),
        true_anomaly_rad=float(generator.uniform(0.0, 2.0 * np.pi)),
        gravitational_parameter=MU_EARTH,
    )
    state = state_from_elements(elements, epoch_s=0.0)
    return CatalogObject(
        object_id=object_id,
        state=state,
        radius_m=float(generator.uniform(0.3, 4.0)),
        covariance_ric=_covariance_for(generator, elements, state),
    )


def _planted_object(
    generator: np.random.Generator,
    primary: CatalogObject,
    object_id: str,
    window_s: float,
    max_attempts: int = 200,
) -> tuple[CatalogObject, PlantedConjunction]:
    for _ in range(max_attempts):
        tca_s = float(generator.uniform(0.08 * window_s, 0.92 * window_s))
        at_tca = propagate(primary.state, tca_s)

        axis = _random_unit(generator)
        angle = float(generator.uniform(np.deg2rad(12.0), np.deg2rad(110.0)))
        speed_factor = float(generator.uniform(0.97, 1.03))
        secondary_velocity = speed_factor * _rotate_about(at_tca.velocity_m_s, axis, angle)
        relative_velocity = secondary_velocity - at_tca.velocity_m_s

        miss_direction = _perpendicular_unit(generator, relative_velocity)
        # Log-uniform so the catalogue spans the whole decision range, from
        # encounters well inside any action threshold to ones that are clearly safe.
        miss_distance = float(np.exp(generator.uniform(np.log(40.0), np.log(3_500.0))))
        secondary_position = at_tca.position_m + miss_distance * miss_direction

        candidate = OrbitState(
            epoch_s=tca_s, position_m=secondary_position, velocity_m_s=secondary_velocity
        )
        try:
            elements = elements_from_state(candidate)
        except ValueError:
            continue
        perigee_altitude = elements.perigee_radius_m - EARTH_RADIUS_M
        apogee_altitude = elements.apogee_radius_m - EARTH_RADIUS_M
        if perigee_altitude < _MIN_PERIGEE_ALTITUDE_M or apogee_altitude > _MAX_APOGEE_ALTITUDE_M:
            continue

        at_epoch = propagate(candidate, -tca_s)
        catalog_object = CatalogObject(
            object_id=object_id,
            state=at_epoch,
            radius_m=float(generator.uniform(0.5, 3.0)),
            covariance_ric=_covariance_for(generator, elements_from_state(at_epoch), at_epoch),
        )
        truth = PlantedConjunction(
            object_id=object_id,
            tca_s=tca_s,
            miss_distance_m=miss_distance,
            relative_speed_m_s=float(np.linalg.norm(relative_velocity)),
        )
        return catalog_object, truth
    raise RuntimeError(f"could not construct a planted conjunction for {object_id}")


def generate_catalog(
    count: int = 240,
    planted: int = 6,
    window_s: float = 86_400.0,
    seed: int = 20260731,
    primary: CatalogObject | None = None,
) -> SyntheticCatalog:
    """Generate a synthetic catalogue with a known number of planted conjunctions.

    Args:
        count: Total number of secondary objects, planted ones included.
        planted: Number of secondaries constructed to conjunct with the primary.
        window_s: Screening window the planted times of closest approach fall in.
        seed: Seed for the generator.
        primary: Primary object, or None to use :func:`default_primary`.

    Returns:
        The catalogue and the ground truth for its planted encounters.
    """
    if planted > count:
        raise ValueError("planted count cannot exceed the total object count")
    generator = np.random.default_rng(seed)
    reference = primary or default_primary()

    secondaries: list[CatalogObject] = []
    truths: list[PlantedConjunction] = []
    for index in range(planted):
        catalog_object, truth = _planted_object(
            generator, reference, f"PLANTED-{index + 1:02d}", window_s
        )
        secondaries.append(catalog_object)
        truths.append(truth)
    for index in range(count - planted):
        secondaries.append(_background_object(generator, f"DEBRIS-{index + 1:04d}"))

    return SyntheticCatalog(
        primary=reference,
        secondaries=tuple(secondaries),
        planted=tuple(truths),
        seed=seed,
        window_s=window_s,
    )


def catalog_element_table(catalog: SyntheticCatalog) -> Matrix:
    """Return one row per secondary with semi-major axis, eccentricity, and inclination."""
    rows = [
        [
            item.elements.semi_major_axis_m,
            item.elements.eccentricity,
            item.elements.inclination_rad,
        ]
        for item in catalog.secondaries
    ]
    return np.array(rows, dtype=np.float64)
