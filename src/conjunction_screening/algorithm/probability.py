"""Two-dimensional probability of collision.

All four implementations answer the same question: given a bivariate Gaussian on
the encounter plane with mean at the projected miss vector and covariance equal
to the projected combined covariance, what is the mass of that density inside a
disc of the combined hard body radius centred on the primary?

    Pc = (1 / (2 pi sx sy)) * integral over the disc of
         exp(-0.5 * (((x - mx) / sx)^2 + ((y - my) / sy)^2)) dx dy

written here in the principal axes of the covariance, where the density
separates. The four methods differ in how the integral is evaluated:

* ``FosterMethod`` integrates in polar coordinates over the disc with adaptive
  quadrature, which is the formulation of Foster and Estes (1992).
* ``AlfanoMethod`` performs the inner integral analytically with the error
  function and applies Simpson's rule to what is left, following Alfano (2005).
* ``ChanMethod`` evaluates Chan's convergent series, which is exact for a
  circular covariance and uses an equal-area circle otherwise.
* ``MonteCarloMethod`` samples the three-dimensional relative position error and
  counts how many straight-line trajectories pass within the hard body radius. It
  exercises the projection as well as the integral and is the reference the other
  three are checked against.

Foster and Alfano are independent formulations of the same integral, so their
agreement is a genuine cross validation rather than a restatement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from scipy.integrate import dblquad
from scipy.special import erf, erfc, gammaln

from conjunction_screening.model.encounter import (
    EncounterGeometry,
    PrincipalForm,
    principal_axis_form,
)

__all__ = [
    "ALFANO",
    "CHAN",
    "FOSTER",
    "MONTE_CARLO",
    "AlfanoMethod",
    "ChanMethod",
    "FosterMethod",
    "MonteCarloMethod",
    "ProbabilityMethod",
    "ProbabilityResult",
]

FOSTER: Final[str] = "foster"
ALFANO: Final[str] = "alfano"
CHAN: Final[str] = "chan"
MONTE_CARLO: Final[str] = "monte-carlo"

_SQRT_TWO: Final[float] = float(np.sqrt(2.0))
_SQRT_TWO_PI: Final[float] = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True, slots=True)
class ProbabilityResult:
    """One probability estimate with its own accuracy statement.

    Attributes:
        method: Identifier of the method that produced the value.
        value: Probability of collision, dimensionless, in [0, 1].
        error_estimate: The method's own bound or standard error on the value.
            For the quadrature methods it is an absolute quadrature error
            estimate; for Monte Carlo it is the binomial standard error.
        converged: Whether the method met its own tolerance. A false value must
            never be pinned in a regression test.
        detail: Short description of how the estimate terminated.
    """

    method: str
    value: float
    error_estimate: float
    converged: bool
    detail: str


class ProbabilityMethod(Protocol):
    """A way of evaluating the probability of collision for one encounter."""

    @property
    def name(self) -> str:
        """Identifier recorded alongside every result this method produces."""
        ...

    def probability(self, encounter: EncounterGeometry) -> ProbabilityResult:
        """Return the probability of collision for ``encounter``."""
        ...


@dataclass(frozen=True, slots=True)
class FosterMethod:
    """Foster and Estes polar quadrature over the hard body disc.

    Attributes:
        absolute_tolerance: Absolute tolerance passed to the adaptive quadrature.
        relative_tolerance: Relative tolerance passed to the adaptive quadrature.
    """

    absolute_tolerance: float = 1e-14
    relative_tolerance: float = 1e-11

    @property
    def name(self) -> str:
        """Identifier of this method."""
        return FOSTER

    def probability(self, encounter: EncounterGeometry) -> ProbabilityResult:
        """Evaluate the disc integral in polar coordinates."""
        form = principal_axis_form(encounter)
        sigma_x, sigma_y = form.sigma_x_m, form.sigma_y_m
        mean_x, mean_y = form.mean_x_m, form.mean_y_m
        radius = form.radius_m

        def integrand(angle: float, distance: float) -> float:
            x = distance * np.cos(angle) - mean_x
            y = distance * np.sin(angle) - mean_y
            return float(distance * np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2)))

        integral, error = dblquad(
            integrand,
            0.0,
            radius,
            0.0,
            2.0 * np.pi,
            epsabs=self.absolute_tolerance,
            epsrel=self.relative_tolerance,
        )
        normaliser = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
        value = float(np.clip(integral * normaliser, 0.0, 1.0))
        scaled_error = float(error) * normaliser
        converged = scaled_error <= max(
            self.absolute_tolerance, self.relative_tolerance * max(value, 0.0)
        )
        return ProbabilityResult(
            method=FOSTER,
            value=value,
            error_estimate=scaled_error,
            converged=converged,
            detail=f"adaptive polar quadrature, estimated absolute error {scaled_error:.3e}",
        )


@dataclass(frozen=True, slots=True)
class AlfanoMethod:
    """Alfano's reduction of the disc integral to a single smooth integral.

    Integrating the density across the chord of the disc at fixed abscissa gives a
    difference of error functions, leaving

        Pc = integral over x in [-R, R] of
             (1 / (sqrt(2 pi) sx)) exp(-(x - mx)^2 / (2 sx^2))
             * 0.5 * (erf((h - my) / (sy sqrt 2)) + erf((h + my) / (sy sqrt 2))) dx

    with ``h = sqrt(R^2 - x^2)``. That integrand has an infinite derivative at the
    two endpoints, where Simpson's rule loses its fourth-order accuracy. The
    substitution ``x = R sin t`` maps the interval to ``[-pi/2, pi/2]`` and makes
    the integrand analytic, restoring the rate. The result is still Alfano's
    formulation; only the abscissa is changed.

    Attributes:
        relative_tolerance: Convergence threshold on the change between
            successive Simpson refinements, relative to the newer value.
        initial_intervals: Even number of Simpson intervals at the first level.
        max_intervals: Cap on the interval count before the solve is declared
            non-converged.
    """

    relative_tolerance: float = 1e-11
    initial_intervals: int = 64
    max_intervals: int = 1 << 18

    @property
    def name(self) -> str:
        """Identifier of this method."""
        return ALFANO

    def probability(self, encounter: EncounterGeometry) -> ProbabilityResult:
        """Evaluate the reduced integral by repeated Simpson refinement."""
        form = principal_axis_form(encounter)
        intervals = max(self.initial_intervals, 2)
        previous = _alfano_simpson(form, intervals)
        while intervals < self.max_intervals:
            intervals *= 2
            current = _alfano_simpson(form, intervals)
            change = abs(current - previous)
            if change <= self.relative_tolerance * max(current, np.finfo(float).tiny):
                value = float(np.clip(current, 0.0, 1.0))
                return ProbabilityResult(
                    method=ALFANO,
                    value=value,
                    error_estimate=change,
                    converged=True,
                    detail=f"Simpson's rule converged at {intervals} intervals",
                )
            previous = current
        value = float(np.clip(previous, 0.0, 1.0))
        return ProbabilityResult(
            method=ALFANO,
            value=value,
            error_estimate=float("nan"),
            converged=False,
            detail=f"Simpson's rule did not converge by {self.max_intervals} intervals",
        )


def _alfano_simpson(form: PrincipalForm, intervals: int) -> float:
    """Apply composite Simpson's rule to the Alfano integrand at a fixed resolution."""
    nodes = np.linspace(-0.5 * np.pi, 0.5 * np.pi, intervals + 1)
    values = _alfano_integrand(form, nodes)
    weights = np.ones(intervals + 1, dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    step = np.pi / intervals
    return float(step / 3.0 * float(np.dot(weights, values)))


def _alfano_integrand(form: PrincipalForm, nodes: np.ndarray) -> np.ndarray:
    radius = form.radius_m
    sigma_x, sigma_y = form.sigma_x_m, form.sigma_y_m
    abscissa = radius * np.sin(nodes)
    chord = radius * np.cos(nodes)
    density = np.exp(-0.5 * ((abscissa - form.mean_x_m) / sigma_x) ** 2) / (_SQRT_TWO_PI * sigma_x)
    swept = _swept_mass(chord, form.mean_y_m, sigma_y)
    return np.asarray(chord * density * swept, dtype=np.float64)


def _swept_mass(chord: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Return the Gaussian mass between ``-chord`` and ``+chord`` about ``mean``.

    The expression is even in ``mean``, so the magnitude is used. Two algebraically
    equal forms are available and they fail in opposite regimes. The ``erf`` form
    adds two values of magnitude at most one, which cancels catastrophically once
    the mean is several standard deviations outside the chord, because both terms
    saturate at plus and minus one. The ``erfc`` form subtracts two small tail
    values and stays accurate there, but loses digits when both arguments are near
    zero. Switching at one standard deviation keeps the relative error near
    machine precision across the whole range, which matters because a screening
    run routinely evaluates probabilities below 1e-12.
    """
    offset = abs(mean) / (sigma * _SQRT_TWO)
    scaled = chord / (sigma * _SQRT_TWO)
    if offset < 1.0:
        return np.asarray(0.5 * (erf(scaled + offset) + erf(scaled - offset)), dtype=np.float64)
    return np.asarray(0.5 * (erfc(offset - scaled) - erfc(offset + scaled)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ChanMethod:
    """Chan's convergent series for the probability of collision.

    Scaling the encounter plane by the two principal standard deviations turns
    the Gaussian into a standard bivariate normal and the hard body circle into an
    ellipse. Chan replaces that ellipse with the circle of equal area, radius
    ``R / sqrt(sx sy)``, after which the mass inside is a non-central chi-square
    tail with two degrees of freedom:

        u = R^2 / (sx sy),  v = (mx / sx)^2 + (my / sy)^2

        Pc = sum over m >= 0 of  Poisson(m; v/2) * P(Poisson(u/2) > m)

    The equal-area substitution is exact when ``sx == sy`` and approximate
    otherwise, so this method is used as a cross check rather than as the primary
    result. Each term is evaluated from a positive series rather than by
    subtracting nearly equal quantities, which keeps small probabilities accurate.

    Attributes:
        relative_tolerance: The outer sum stops when the remaining Poisson mass
            is below this fraction of the running total.
        max_terms: Cap on outer terms before the series is declared
            non-converged.
    """

    relative_tolerance: float = 1e-13
    max_terms: int = 2_000

    @property
    def name(self) -> str:
        """Identifier of this method."""
        return CHAN

    def probability(self, encounter: EncounterGeometry) -> ProbabilityResult:
        """Sum the series to the requested tolerance."""
        form = principal_axis_form(encounter)
        u = form.radius_m**2 / (form.sigma_x_m * form.sigma_y_m)
        v = (form.mean_x_m / form.sigma_x_m) ** 2 + (form.mean_y_m / form.sigma_y_m) ** 2
        half_u = 0.5 * u
        half_v = 0.5 * v

        total = 0.0
        outer_weight = float(np.exp(-half_v))
        remaining = 1.0 - outer_weight
        bound = float("inf")
        for term in range(self.max_terms):
            tail = _poisson_upper_tail(half_u, term)
            total += outer_weight * tail
            # The inner tail decreases with the term index, so the mass still to
            # be summed multiplied by the current tail bounds the truncation error.
            bound = remaining * tail
            if bound <= self.relative_tolerance * max(total, np.finfo(float).tiny):
                value = float(np.clip(total, 0.0, 1.0))
                return ProbabilityResult(
                    method=CHAN,
                    value=value,
                    error_estimate=bound,
                    converged=True,
                    detail=f"series converged after {term + 1} terms",
                )
            outer_weight *= half_v / (term + 1)
            remaining = max(remaining - outer_weight, 0.0)
        return ProbabilityResult(
            method=CHAN,
            value=float(np.clip(total, 0.0, 1.0)),
            error_estimate=bound,
            converged=False,
            detail=f"series did not converge in {self.max_terms} terms",
        )


def _poisson_upper_tail(rate: float, index: int, max_terms: int = 4_000) -> float:
    """Return ``P(N > index)`` for ``N`` Poisson with mean ``rate``.

    Summed forward from term ``index + 1`` so that every contribution is positive.
    The alternative, one minus the lower partial sum, cancels catastrophically
    when the rate is small, which is exactly the regime of a real conjunction
    where the hard body radius is far smaller than the position uncertainty.
    """
    if rate <= 0.0:
        return 0.0
    log_term = -rate + (index + 1) * np.log(rate) - float(_log_factorial(index + 1))
    if log_term < -745.0:
        return 0.0
    term = float(np.exp(log_term))
    total = term
    for step in range(index + 2, index + 2 + max_terms):
        term *= rate / step
        total += term
        if term <= 1e-18 * total:
            break
    return total


def _log_factorial(value: int) -> float:
    return float(gammaln(value + 1))


@dataclass(frozen=True, slots=True)
class MonteCarloMethod:
    """Direct simulation of the encounter under the linear relative motion model.

    Position errors are drawn from the three-dimensional combined covariance at
    the time of closest approach. Each draw defines a straight-line relative
    trajectory whose closest approach to the primary is the component of the
    perturbed relative position perpendicular to the relative velocity. Counting
    the draws that pass within the hard body radius therefore tests the encounter
    plane construction and the projection of the covariance as well as the value
    of the integral.

    Attributes:
        samples: Number of draws.
        seed: Seed for the generator, so a run is reproducible.
        block: Draws per batch, which bounds peak memory.
        minimum_hits: Hits below which the estimate is reported as not converged.
            A binomial estimate from a handful of hits has a relative standard
            error near one over the square root of the hit count, so an estimate
            built on fewer hits than this carries no useful precision.
    """

    samples: int = 2_000_000
    seed: int = 20260731
    block: int = 250_000
    minimum_hits: int = 25

    @property
    def name(self) -> str:
        """Identifier of this method."""
        return MONTE_CARLO

    def probability(self, encounter: EncounterGeometry) -> ProbabilityResult:
        """Estimate the probability by sampling, and report the binomial standard error."""
        generator = np.random.default_rng(self.seed)
        factor = _covariance_square_root(encounter.relative_covariance.matrix)
        direction = np.asarray(encounter.relative_velocity_m_s, dtype=np.float64)
        direction = direction / float(np.linalg.norm(direction))
        mean = np.asarray(encounter.relative_position_m, dtype=np.float64)
        radius = encounter.hard_body_radius_m

        hits = 0
        drawn = 0
        while drawn < self.samples:
            count = min(self.block, self.samples - drawn)
            normal = generator.standard_normal((count, 3))
            offsets = mean + normal @ factor.T
            along = offsets @ direction
            perpendicular = offsets - along[:, None] * direction[None, :]
            hits += int(np.count_nonzero(np.linalg.norm(perpendicular, axis=1) <= radius))
            drawn += count

        estimate = hits / self.samples
        standard_error = float(np.sqrt(max(estimate * (1.0 - estimate), 0.0) / self.samples))
        return ProbabilityResult(
            method=MONTE_CARLO,
            value=float(estimate),
            error_estimate=standard_error,
            converged=hits >= self.minimum_hits,
            detail=f"{hits} hits in {self.samples} draws, standard error {standard_error:.3e}",
        )


def _covariance_square_root(matrix: np.ndarray) -> np.ndarray:
    """Return a matrix ``L`` with ``L L^T`` equal to ``matrix``.

    An eigendecomposition is used rather than a Cholesky factorisation so that a
    covariance that is positive semi-definite but singular, which arises whenever
    one direction carries no uncertainty, is still handled.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    clipped = np.clip(eigenvalues, 0.0, None)
    return np.asarray(eigenvectors * np.sqrt(clipped), dtype=np.float64)
