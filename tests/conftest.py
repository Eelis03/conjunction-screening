"""Shared fixtures.

The catalogue and screening fixtures are session scoped because generating a
catalogue and screening it is the most expensive thing the suite does, and every
tier of the suite needs the same one.
"""

from __future__ import annotations

import pytest

from conjunction_screening.algorithm.close_approach import CloseApproachSettings
from conjunction_screening.model.encounter import EncounterGeometry, planar_encounter
from conjunction_screening.pipeline.catalog import SyntheticCatalog, generate_catalog
from conjunction_screening.pipeline.screening import ScreeningConfig, ScreeningReport, run_screening

REGRESSION_SEED = 20260731
"""Seed of the catalogue pinned by the regression tests."""

REGRESSION_THRESHOLD_M = 5_000.0
"""Screening threshold pinned by the regression tests."""


@pytest.fixture(scope="session")
def close_approach_settings() -> CloseApproachSettings:
    """Search settings shared by the close approach tests."""
    return CloseApproachSettings(threshold_m=REGRESSION_THRESHOLD_M)


@pytest.fixture(scope="session")
def conjunction_catalog() -> SyntheticCatalog:
    """A catalogue in which every secondary is a planted conjunction.

    Used by the filter safety tests, where every object must survive the cascade.
    """
    return generate_catalog(count=12, planted=12, window_s=86_400.0, seed=4021)


@pytest.fixture(scope="session")
def mixed_catalog() -> SyntheticCatalog:
    """A catalogue of background objects with a few planted conjunctions."""
    return generate_catalog(count=60, planted=6, window_s=86_400.0, seed=REGRESSION_SEED)


@pytest.fixture(scope="session")
def regression_catalog() -> SyntheticCatalog:
    """The catalogue whose screening run the regression tests pin."""
    return generate_catalog(count=120, planted=8, window_s=86_400.0, seed=REGRESSION_SEED)


@pytest.fixture(scope="session")
def regression_report(regression_catalog: SyntheticCatalog) -> ScreeningReport:
    """The pinned screening run."""
    return run_screening(regression_catalog, ScreeningConfig.for_threshold(REGRESSION_THRESHOLD_M))


@pytest.fixture
def isotropic_encounter() -> EncounterGeometry:
    """A circular in-plane covariance, for which Chan's series is exact."""
    return planar_encounter(
        miss_distance_m=180.0,
        sigma_x_m=260.0,
        sigma_y_m=260.0,
        hard_body_radius_m=12.0,
    )


@pytest.fixture
def elongated_encounter() -> EncounterGeometry:
    """An in-plane covariance with a twenty to one aspect ratio and a rotated frame."""
    return planar_encounter(
        miss_distance_m=320.0,
        sigma_x_m=2_000.0,
        sigma_y_m=100.0,
        hard_body_radius_m=14.0,
        orientation_rad=0.7,
    )
