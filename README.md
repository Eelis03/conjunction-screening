# Conjunction Screening

Conjunction filtering and probability of collision using the Foster and Alfano methods.

[![CI](https://github.com/Eelis03/conjunction-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/conjunction-screening/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library screens a catalogue of orbiting objects against a primary satellite and
reports the probability of collision for every conjunction it finds. It applies the
three-filter cascade of Hoots, Crawford, and Roehrich to discard pairs that cannot
conjunct, refines the surviving pairs to a time of closest approach, propagates the
position covariances to that time, and evaluates the two-dimensional probability of
collision by the Foster method, the Alfano method, and Chan's series so that the three can
be compared. It is written for anyone building or reviewing a conjunction assessment
pipeline who needs the arithmetic to be checkable rather than taken on trust.

## Problem

A satellite operator receives a catalogue of tracked objects and has to decide, before each
of them passes, whether to spend propellant avoiding it. Three things make that hard.

The first is cost. Screening one primary against a catalogue means examining every pair, and
a full close approach search over a day of orbital motion is far too expensive to run on all
of them. Filters are needed, and a filter that discards a real conjunction is worse than no
filter at all, so each one has to come with an argument that it cannot.

The second is that the quantity the decision rests on is a probability, not a distance. A
miss distance of two kilometres is not safe if the covariance is ten kilometres wide, and a
miss distance of two hundred metres is not dangerous if the covariance is ten metres wide.
Computing the probability requires propagating the covariance of both objects to the time of
closest approach, transforming it into the encounter plane, and integrating a bivariate
Gaussian over a disc.

The third is that the probability of collision behaves in a way that is easy to misread. It
is not monotonic in the size of the covariance. Inflating the covariance raises the
probability up to a point and lowers it after that, so an object that is poorly tracked can
report a smaller probability than the same object tracked well. A screening system that
treats a small probability as evidence of safety, without also asking how large the
covariance was, will dismiss exactly the conjunctions it understands least.

## Approach

Filtering follows the published cascade of Hoots, Crawford, and Roehrich (1984), applied in
their order and from cheapest to most expensive. The perigee and apogee filter compares the
radial shells the two orbits occupy and rejects a pair when the gap between the shells
exceeds the screening threshold, which is exact by the reverse triangle inequality. The
orbit path filter bounds the minimum distance between the two orbit paths treated as static
curves. The time filter maps the parts of each path that can produce a close approach into
time intervals through Kepler's equation and rejects a pair whose intervals never coincide.

The orbit path filter is where the conservativeness argument has to be made carefully. The
separation between two points on the two paths is a Lipschitz function of the two true
anomalies, with a constant that has a closed form, so a sampled separation can be turned
into a rigorous lower bound over a whole cell of anomaly space. A branch and bound over
those cells either finds a sample below the threshold, in which case the pair passes, or
prunes every cell, in which case the pair is proved safe. Both budgets that stop the search
early return a pass, so a budget exhaustion can never become a missed conjunction.

Close approach determination refines a coarse sweep with Brent's method applied to the
product of the relative position and the relative velocity, which vanishes at every extremum
of the range and stays smooth through it. The coarse step is derived from the screening
threshold and the relative speed rather than chosen, so no approach that reaches the
threshold falls between two samples.

Covariances are carried from the RIC frame at the catalogue epoch, into the inertial frame,
forward with the state transition matrix of the two-body flow, and then into the encounter
plane normal to the relative velocity. That last projection is what makes the
two-dimensional formulation valid: under linear relative motion the secondary crosses the
plane in a straight line, so a three-dimensional collision question becomes a
two-dimensional one about a disc.

The probability integral is evaluated three ways. Foster and Estes (1992) integrate in polar
coordinates over the disc. Alfano (2005a) performs the inner integral analytically with the
error function and applies Simpson's rule to the remainder. Chan (2008) sums a convergent
series that is exact for a circular covariance and uses an equal-area circle otherwise.
Foster and Alfano share nothing but the reduction to principal axes, so their agreement is a
genuine cross validation. A Monte Carlo estimator that samples the three-dimensional
relative position and counts straight-line passes inside the hard body radius checks the
projection as well as the integral.

The maximum probability over an isotropic scaling of the covariance follows Alfano (2005b),
and is checked against the closed form for a circular covariance. docs/design-notes.md
records the alternatives that were considered and not chosen.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/conjunction-screening.git
cd conjunction-screening
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
from conjunction_screening import generate_catalog, run_screening
from conjunction_screening.analysis.ranking import format_ranking_table, rank_report
from conjunction_screening.pipeline.screening import ScreeningConfig

catalog = generate_catalog(count=240, planted=8, window_s=86_400.0, seed=20260731)
report = run_screening(catalog, ScreeningConfig.for_threshold(5_000.0))

print(report.rejection_counts)
# {'orbit-path': 28, 'perigee-apogee': 201, 'time': 3}

print(format_ranking_table(rank_report(report), limit=3))
# rank  object               tca [s]    miss [m]  v_rel [m/s]  radius [m]           Pc  action
# ----------------------------------------------------------------------------------------------
#    1  PLANTED-05         12353.126       122.5        744.2         6.7   1.4379e-04  act
#    2  PLANTED-06         67252.006       103.5       1670.4         7.8   1.3852e-04  act
#    3  PLANTED-01         57650.359       230.7       2644.7         7.3   1.0201e-05  monitor
# ... 5 further event(s) not shown
```

Runnable examples live in `examples/`:

```bash
uv run python examples/screen_catalog.py
uv run python examples/dilution_study.py
uv run python examples/method_comparison.py
```

Each accepts `--reduced` to run a smaller catalogue, and `--help` to list its options.

## Results

Every number below was produced by the commands shown, on a synthetic catalogue generated
from seed 20260731. No orbital element set is downloaded or embedded.

### Screening run

`uv run python examples/screen_catalog.py`, 240 secondary objects, an 86400 s window, and a
5000 m screening threshold. The catalogue is generated in 0.11 s and screened in 0.18 s.

| Stage | Rejected | Remaining |
| --- | --- | --- |
| perigee and apogee filter | 201 | 39 |
| orbit path filter | 28 | 11 |
| time filter | 3 | 8 |

The eight survivors produced eight conjunction events. The time filter narrowed the close
approach search to 2792.1 s of candidate windows, against the 691200 s that searching the
whole window for all eight survivors would have covered, a reduction by a factor of 247.6.

| Rank | Object | TCA [s] | Miss [m] | Relative speed [m/s] | Combined radius [m] | Pc | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PLANTED-05 | 12353.126 | 122.5 | 744.2 | 6.7 | 1.4379e-04 | act |
| 2 | PLANTED-06 | 67252.006 | 103.5 | 1670.4 | 7.8 | 1.3852e-04 | act |
| 3 | PLANTED-01 | 57650.359 | 230.7 | 2644.7 | 7.3 | 1.0201e-05 | monitor |
| 4 | PLANTED-02 | 21248.427 | 1997.8 | 471.0 | 6.8 | 1.3181e-10 | dismiss |
| 5 | PLANTED-08 | 73496.495 | 907.9 | 786.3 | 5.8 | 8.3734e-19 | dismiss |
| 6 | PLANTED-07 | 32526.929 | 500.9 | 4576.0 | 5.6 | 2.1838e-19 | dismiss |
| 7 | PLANTED-04 | 66519.202 | 1710.7 | 2938.4 | 7.4 | 4.5812e-224 | dismiss |
| 8 | PLANTED-03 | 53295.102 | 3382.9 | 572.3 | 6.4 | 0.0000e+00 | dismiss |

The action thresholds are 1e-4 for act and 1e-7 for monitor. Ranks 5 and 6 show why miss
distance alone is not a ranking: rank 6 has a smaller miss distance than rank 5 and still a
smaller probability, because its combined covariance is wider.

### Foster against Alfano, and against Chan

`uv run python examples/method_comparison.py`, eight encounters spanning miss distances from
50 m to 700 m and in-plane covariance aspect ratios from 1 to 20.

| Case | Miss [m] | R [m] | sigma x [m] | sigma y [m] | Foster | Alfano | Chan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| circular-near | 50.0 | 10.0 | 100.0 | 100.0 | 4.402846e-03 | 4.402846e-03 | 4.402846e-03 |
| circular-mid | 200.0 | 12.0 | 250.0 | 250.0 | 8.361961e-04 | 8.361961e-04 | 8.361961e-04 |
| circular-far | 700.0 | 8.0 | 300.0 | 300.0 | 2.337730e-05 | 2.337730e-05 | 2.337730e-05 |
| elongated-2to1 | 150.0 | 10.0 | 400.0 | 200.0 | 5.823430e-04 | 5.823430e-04 | 5.823948e-04 |
| elongated-5to1 | 300.0 | 12.0 | 1000.0 | 200.0 | 2.626679e-04 | 2.626679e-04 | 2.626916e-04 |
| elongated-20to1 | 200.0 | 15.0 | 2000.0 | 100.0 | 1.260562e-04 | 1.260562e-04 | 1.253716e-04 |
| wide-covariance | 120.0 | 10.0 | 3000.0 | 1500.0 | 1.110036e-05 | 1.110036e-05 | 1.110038e-05 |
| tight-covariance | 80.0 | 9.0 | 60.0 | 40.0 | 4.004085e-03 | 4.004085e-03 | 3.995193e-03 |

Worst relative difference between Foster and Alfano over the eight cases: 4.348e-16, which
is at the level of the double precision representation and well inside the 1e-11 relative
tolerance each quadrature was asked for.

Worst relative difference between Foster and Chan: 5.431e-03, on the tight-covariance case.
Chan agrees with Foster to 5.910e-16 relative on the three circular cases, where its
equal-area substitution is an identity, and departs from it as the aspect ratio grows. That
is the expected behaviour of the approximation, and is why Chan is used here as a cross
check rather than as the primary result.

### Monte Carlo cross check

Same command, 4000000 draws per case. The estimator samples the three-dimensional relative
position from the combined covariance, projects each draw onto the plane normal to the
relative velocity, and counts the draws inside the combined hard body radius, so it tests
the encounter plane construction as well as the integral.

| Case | Foster | Monte Carlo | Standard error | Deviation |
| --- | --- | --- | --- | --- |
| circular-near | 4.402846e-03 | 4.408250e-03 | 3.312e-05 | 0.16 sigma |
| circular-mid | 8.361961e-04 | 8.352500e-04 | 1.444e-05 | 0.07 sigma |
| circular-far | 2.337730e-05 | 1.950000e-05 | 2.208e-06 | 1.76 sigma |
| elongated-2to1 | 5.823430e-04 | 5.890000e-04 | 1.213e-05 | 0.55 sigma |
| elongated-5to1 | 2.626679e-04 | 2.657500e-04 | 8.150e-06 | 0.38 sigma |
| elongated-20to1 | 1.260562e-04 | 1.360000e-04 | 5.831e-06 | 1.71 sigma |
| wide-covariance | 1.110036e-05 | 9.500000e-06 | 1.541e-06 | 1.04 sigma |
| tight-covariance | 4.004085e-03 | 4.030250e-03 | 3.168e-05 | 0.83 sigma |

Every case lies within 1.76 binomial standard errors of the analytic value. The standard
error is computed from the estimate itself, so the check tightens as the sample count grows
rather than being calibrated to the difference that happened to be observed.

### Dilution and maximum probability

`uv run python examples/dilution_study.py`, covariance scaled over five decades.

An isotropic reference case, miss distance 100 m, nominal sigma 500 m in both in-plane
directions, combined hard body radius 10 m:

| Quantity | Value |
| --- | --- |
| Pc at the nominal covariance | 1.960205e-04 |
| maximum Pc found numerically | 3.678810e-03 at a covariance scale of 0.1411 |
| geometric mean sigma at the maximum | 70.5 m |
| closed form R^2 / (e d^2) | 3.678794e-03 |
| closed form d / sqrt(2) | 70.7 m |
| dilution factor, maximum divided by nominal | 18.77 |
| fitted slope of log Pc against log scale, largest decade | -2.0000 |

The numerical maximum agrees with the closed form to 4.3e-6 relative, and the location of
the maximum agrees to 0.3 percent. The fitted slope of -2.0000 confirms the predicted
inverse square asymptote of the falling branch.

The curve is not monotonic. Scaling the covariance from 0.01 to 1000 gives, at selected
scales, the following probabilities:

| Covariance scale | Pc |
| --- | --- |
| 0.0100 | 2.756176e-73 |
| 0.0681 | 6.210928e-04 |
| 0.1778 | 3.356238e-03 |
| 0.4642 | 8.456633e-04 |
| 1.2115 | 1.344053e-04 |
| 8.2540 | 2.934733e-06 |
| 56.2341 | 6.324515e-08 |
| 1000.0000 | 2.000000e-10 |

At a covariance scale of 8.2540 the probability is 2.934733e-06, which is 66.79 times
smaller than the 1.960205e-04 reported at the nominal covariance, even though the miss
distance and the hard body radius are unchanged. That is the dilution region, and it is the
reason a small probability of collision is not by itself a reason to dismiss a conjunction.

The highest-ranked event of the screening run, PLANTED-05, shows the same behaviour on real
pipeline output. Its nominal in-plane covariance is 1554.5 m by 84.2 m, its miss distance
122.5 m, and its probability 1.437928e-04. Scaling that covariance reaches a maximum of
3.458534e-04 at a scale of 0.4272, a dilution factor of 2.41, so the nominal covariance
already sits past the peak. The closed form value of 1.111049e-03 for that geometry is
higher than the achievable maximum here because it applies to a circular covariance, and
inflating an elongated one isotropically explores a different family.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/conjunction_screening/model/constants.py` | Physical constants in SI units |
| `src/conjunction_screening/model/arrays.py` | Read-only array construction and shape checking |
| `src/conjunction_screening/model/state.py` | Orbit states, classical elements, Kepler's equation, the element Jacobian |
| `src/conjunction_screening/model/covariance.py` | Covariance type, definiteness checking, element-space construction |
| `src/conjunction_screening/model/frames.py` | Rotations between the inertial frame and the RIC frame |
| `src/conjunction_screening/model/encounter.py` | Encounter plane basis, projection, principal axis reduction |
| `src/conjunction_screening/algorithm/propagation.py` | Two-body propagation, state transition matrix, covariance propagation |
| `src/conjunction_screening/algorithm/filters.py` | The three-filter cascade and its conservativeness bounds |
| `src/conjunction_screening/algorithm/close_approach.py` | Coarse sweep and Brent refinement of the time of closest approach |
| `src/conjunction_screening/algorithm/probability.py` | Foster, Alfano, Chan, and Monte Carlo behind one Protocol |
| `src/conjunction_screening/algorithm/maximum.py` | Maximum probability over covariance scaling |
| `src/conjunction_screening/pipeline/catalog.py` | Synthetic catalogue generation with planted conjunctions |
| `src/conjunction_screening/pipeline/screening.py` | The screening run and its structured trace |
| `src/conjunction_screening/analysis/ranking.py` | Ranking and the mapping from probability to action |
| `src/conjunction_screening/analysis/dilution.py` | The dilution study |
| `src/conjunction_screening/analysis/comparison.py` | Cross comparison of the probability methods |
| `src/conjunction_screening/analysis/figures.py` | Figure generation, the only module that writes to disk |
| `examples/` | Wiring scripts with no logic of their own |

The layers depend in one direction only. The model layer is pure functions over immutable
values. The algorithm layer takes model values and returns result records, and does no
plotting. The pipeline layer wires the two together and produces a trace. The analysis layer
ranks, compares, and draws. The examples wire the analysis layer to a command line.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

165 tests run in 10.8 s.

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

The property tier covers the safety property that the whole cascade rests on. Catalogues in
which every secondary is a planted conjunction with a known time of closest approach and a
known miss distance are generated, and every filter is required to pass every one of them.
Selectivity is covered separately, with pairs whose geometry can be checked on paper: two
circular orbits 800 km apart for the perigee and apogee filter, a circle and a perpendicular
ellipse whose paths stay 17.5 km apart for the orbit path filter, and two equal-period
circles crossing a quarter of a revolution out of phase for the time filter. The Lipschitz
constant the orbit path filter depends on is checked against a finely sampled numerical
derivative, because a filter whose bound is not a bound could discard a real conjunction.

Other invariants covered: the state transition matrix is symplectic; the time of closest
approach has zero relative range rate; miss distance is symmetric under swapping the two
objects; the encounter plane projection preserves the magnitude of a perpendicular relative
position; the covariance stays symmetric and positive semi-definite through every stage of
the chain; Foster and Alfano agree; Chan is exact for a circular covariance and departs
monotonically as the aspect ratio grows; probability falls to zero with miss distance and
rises towards one with hard body radius; the dilution curve rises then falls and the maximum
probability calculation finds the peak; and a Monte Carlo estimate agrees with the analytic
value.

Two rules govern the tolerances. Only values from a converged solve are pinned. A close
approach whose refinement did not converge is excluded from the report before it can reach a
regression test, and the regression module asserts that every pinned event converged,
because the state of a non-converged iteration depends on the order a floating point
reduction ran in and differs between machines. Every tolerance is derived from the
measurement rather than from an observed error. The residual range rate is bounded by the
curvature of the range multiplied by the root find tolerance. The agreement between a
computed time of closest approach and the time an encounter was constructed around uses the
propagator round trip error, measured inside the test, divided by the relative speed. The
symplectic residual is bounded in terms of the square of the largest entry of the
non-dimensional transition matrix, so the same expression works from a one minute to a one
day propagation. The Monte Carlo comparison uses four binomial standard errors computed from
the estimate. A separate regression test measures the tightest margin of any filter decision
in the recorded run and requires it to be a substantial fraction of the threshold, so the
pinned counts cannot be resting on a comparison that a rounding difference could flip.

## References

Methods:

- Hoots, F. R., Crawford, L. L., and Roehrich, R. L. "An Analytic Method to Determine Future
  Close Approaches Between Satellites." Celestial Mechanics, Vol. 33, No. 2, 1984,
  pp. 143 to 158. DOI [10.1007/BF01234152](https://doi.org/10.1007/BF01234152). Source of
  the three-filter cascade and of its application order.
- Foster, J. L., and Estes, H. S. "A Parametric Analysis of Orbital Debris Collision
  Probability and Maneuver Rate for Space Vehicles." NASA JSC-25898, NASA Lyndon B. Johnson
  Space Center, August 1992. Stable record:
  [Stanford SearchWorks 13354320](https://searchworks.stanford.edu/view/13354320). Source of
  the polar quadrature formulation of the two-dimensional probability of collision.
- Alfano, S. "A Numerical Implementation of Spherical Object Collision Probability." The
  Journal of the Astronautical Sciences, Vol. 53, No. 1, 2005, pp. 103 to 109.
  DOI [10.1007/BF03546397](https://doi.org/10.1007/BF03546397). Source of the reduction of
  the disc integral to a single integral of error functions evaluated by Simpson's rule.
- Alfano, S. "Relating Position Uncertainty to Maximum Conjunction Probability." The Journal
  of the Astronautical Sciences, Vol. 53, No. 2, 2005, pp. 193 to 205.
  DOI [10.1007/BF03546350](https://doi.org/10.1007/BF03546350). Source of the maximum
  probability over covariance scaling and of the closed form used as its reference.
- Chan, F. K. "Spacecraft Collision Probability." The Aerospace Press and the American
  Institute of Aeronautics and Astronautics, 2008. ISBN 978-1-884989-18-6.
  DOI [10.2514/4.989186](https://doi.org/10.2514/4.989186). Source of the convergent series
  and of the equal-area circle substitution it rests on.

Dependencies:

- [numpy](https://numpy.org/) 2.0 or later. Array arithmetic, symmetric eigendecomposition,
  and the seeded random number generator used by the catalogue and the Monte Carlo
  estimator. BSD 3-Clause licence.
- [scipy](https://scipy.org/) 1.14 or later. Adaptive two-dimensional quadrature for the
  Foster method, Brent root finding for the time of closest approach, bounded scalar
  minimisation for the maximum probability, and the error and log gamma functions.
  BSD 3-Clause licence.
- [matplotlib](https://matplotlib.org/) 3.9 or later. Figure generation in the analysis
  layer. Matplotlib licence, a BSD-compatible licence derived from the Python Software
  Foundation licence.
- [pytest](https://docs.pytest.org/) 8.3 or later, development only. Test runner.
  MIT licence.
- [ruff](https://docs.astral.sh/ruff/) 0.8 or later, development only. Linter and import
  sorter. MIT licence.
- [mypy](https://mypy-lang.org/) 1.13 or later, development only. Static type checker.
  MIT licence.

## License

Released under the MIT license. See [LICENSE](LICENSE).
