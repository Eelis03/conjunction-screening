"""Physical constants used throughout the library.

Values are WGS-84 / EGM-96 and are stated in SI base units. Every length in this
library is a metre, every time a second, and every angle a radian. Kilometres
appear only in report formatting.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "EARTH_RADIUS_M",
    "J2_EARTH",
    "MU_EARTH",
]

MU_EARTH: Final[float] = 3.986004418e14
"""Earth gravitational parameter in m^3 / s^2 (EGM-96)."""

EARTH_RADIUS_M: Final[float] = 6378137.0
"""Earth equatorial radius in m (WGS-84)."""

J2_EARTH: Final[float] = 1.08262668e-3
"""Second zonal harmonic coefficient. Reported for context only.

The propagator in this library is two-body Keplerian. J2 is not applied, because
the filter cascade proves its conservativeness from the assumption that the
orbital elements are constant over the screening window. See docs/design-notes.md.
"""
