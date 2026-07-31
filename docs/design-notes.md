# Design notes for Conjunction Screening

## Method selection

### The filter cascade

The three filters and their order come from Hoots, Crawford, and Roehrich, "An Analytic
Method to Determine Future Close Approaches Between Satellites", Celestial Mechanics 33(2),
1984, DOI 10.1007/BF01234152. The order is the published one and it is also the order of
increasing cost: a comparison of two radial shells, then a search over a two-dimensional
space of true anomalies, then a conversion of arcs into time intervals through Kepler's
equation.

The property that makes a cascade usable is that no filter may reject a pair that could
conjunct. Each of the three is implemented so that its rejection is backed by an inequality
that holds for all true anomalies and all times in the window, not by a sampled minimum.

The perigee and apogee filter is exact. Each object is confined for all time to the radial
shell between its perigee and apogee radius, and the reverse triangle inequality gives
`|r1 - r2| >= abs(|r1| - |r2|)`, so a gap between disjoint shells is a lower bound on the
separation. No approximation enters.

The orbit path filter needs more care, because the minimum distance between two orbit paths
has no useful closed form. The implementation uses a Lipschitz branch and bound. The
separation `d(nu1, nu2)` between a point on each path is Lipschitz in each argument with
constant `max |dr / dnu|`, and for a Keplerian orbit that maximum has the closed form
`a (1 + e) sqrt(1 + (e / (1 - e))^2)`, attained at apogee. Over a cell of half width `w` in
both arguments the separation therefore cannot fall below `d(centre) - (L1 + L2) w`. The
search either finds a cell centre at or below the threshold, which proves the pair can
approach, or prunes every cell, which proves it cannot. Both of the budgets that stop the
search early, the cell count and the pad floor, return a pass, so budget exhaustion degrades
selectivity and never safety. The bound itself is checked in the test suite against a finely
sampled numerical derivative, because a Lipschitz constant that is not one would silently
convert the filter into a source of missed conjunctions.

The time filter follows the same discipline. Both paths are sampled uniformly in true
anomaly, and a sample is marked when the nearest sample on the other path lies within the
threshold plus a pad of `L1 dnu1 / 2 + L2 dnu2 / 2`, which is exactly the worst error the
sampling can introduce. Marked runs are widened by a full sample spacing on each side, more
than covering the half spacing between a true anomaly and its nearest sample, so every true
anomaly at which a conjunction can occur lies inside an arc. Arcs become periodic families of
time intervals through Kepler's equation, and a pair is rejected only when no interval of one
object intersects any interval of the other inside the window. The construction treats the
two arc sets independently rather than tracking which arc pairs with which, which
over-approximates the at-risk set and is therefore the safe direction.

### Close approach determination

The search works on `g(t) = dr . dv` rather than on the range. The range has a square root in
it and is flat near its minimum, so a minimiser applied to it converges slowly and loses
precision; `g` is smooth, crosses zero transversally at every range extremum, and its sign
change from negative to positive identifies a minimum rather than a maximum. Brent's method
is applied inside each bracketed sign change with an absolute tolerance of 1e-9 s.

The coarse step is derived rather than chosen. Over one step of length `h` the range can
change by at most `v h`, so a step of `0.25 D / v` for a threshold `D` places several samples
inside every approach that reaches the threshold. The speed used is the sum of the two
orbital speeds, which bounds the relative speed for all time.

### Covariance handling

Covariances are quoted in the RIC frame at the object epoch, which is how a conjunction data
message carries them, and are propagated to the time of closest approach with the state
transition matrix of the two-body flow, which is the first order or linear method every
operational conjunction assessment system uses. The transition matrix is obtained by central
differences of the analytic propagator with a relative step of 1e-6, which balances a
truncation error of order `h^2` against a round-off error of order `eps / h` and gives about
1e-10 relative accuracy per entry. Because the underlying flow is Hamiltonian, the matrix
must be symplectic, and that identity is checked in the test suite after non-dimensionalising
the matrix so the check compares like with like.

The projection into the encounter plane is what makes the two-dimensional probability valid.
At the time of closest approach the relative position is perpendicular to the relative
velocity, so it already lies in the plane and the projection preserves its magnitude. That is
asserted in the test suite with a tolerance derived from the residual range rate rather than
from an observed error.

### The probability integral

Foster and Estes, NASA JSC-25898, 1992, integrate the combined Gaussian in polar coordinates
over the hard body disc. That is the primary method here, evaluated with adaptive quadrature
at a relative tolerance of 1e-11.

Alfano, Journal of the Astronautical Sciences 53(1), 2005, DOI 10.1007/BF03546397, integrates
the density across the chord of the disc analytically, leaving a single integral of error
functions that Simpson's rule handles. The two methods share only the reduction to the
principal axes of the in-plane covariance, so their agreement is a real cross check. The
implementation differs from the published one in one respect: the integrand has an infinite
derivative at the two endpoints where the chord vanishes, which costs Simpson's rule its
fourth-order rate, so the substitution `x = R sin t` is applied to make the integrand
analytic on the transformed interval. That changes the abscissa and not the formulation.

Two numerical points in that integrand were worth the effort. The mass swept across the chord
is written as a difference of complementary error functions when the miss vector lies more
than one standard deviation outside the chord, and as a sum of error functions when it does
not. The two forms are algebraically identical and fail in opposite regimes: the error
function form loses all its digits once both terms saturate at plus and minus one, which
happens routinely for the small probabilities a screening run produces. Before that change,
the Simpson refinement on a probability of 6e-13 stalled at a relative change of 3e-10 and
was reported as not converged; afterwards it converges at machine precision.

Chan, "Spacecraft Collision Probability", The Aerospace Press, 2008, DOI 10.2514/4.989186,
scales the plane by the two principal standard deviations, replaces the resulting ellipse
with a circle of equal area, and evaluates the remaining non-central chi-square tail as a
convergent series. It is exact when the covariance is circular and approximate otherwise,
which the test suite records in both directions: agreement to 5.9e-16 relative for a circular
covariance, and a disagreement that grows monotonically with the aspect ratio, reaching
5.4e-3 on the cases screened here. Chan is therefore reported as a cross check rather than as
a primary result. The series is summed from positive terms only. The natural expression for
each term is one minus a partial sum of a Poisson series, which cancels catastrophically when
the ratio of hard body radius to covariance scale is small, and that ratio is small in every
real conjunction.

The Monte Carlo estimator samples the three-dimensional relative position from the combined
covariance and projects each draw onto the plane normal to the relative velocity, rather than
sampling the two-dimensional plane distribution directly. Sampling the plane would test only
the quadrature; sampling in three dimensions and projecting also tests the covariance square
root, the plane construction, and the projection, which is most of the pipeline between the
close approach solution and the probability.

### Maximum probability and dilution

The maximum over an isotropic scaling of the covariance follows Alfano, Journal of the
Astronautical Sciences 53(2), 2005, DOI 10.1007/BF03546350. The search runs Brent's bounded
method in the base ten logarithm of the scale, where the curve is close to symmetric and
spans several decades.

The closed form `R^2 / (e d^2)` at `s = d / sqrt(2)` is used as the analytic reference. It is
the maximum over circular covariances and not a bound over all of them: shrinking one
principal axis concentrates the same probability mass into a narrower band, so an elongated
covariance whose wide axis lies along the miss vector can exceed it. The library reports the
closed form alongside the numerical peak and labels it as the circular reference, and the
test suite pins both facts, that a circular covariance reaches it and that an elongated one
can pass it.

## Rejected alternatives

### J2 secular propagation instead of pure two-body

Adding the secular rates of the node, the argument of perigee, and the mean anomaly under the
second zonal harmonic would make the propagation considerably more realistic. It was not
done, and the reason is the filter cascade rather than the effort.

The orbit path filter and the time filter both treat each orbit as a fixed curve in inertial
space over the whole screening window. That is exact under two-body motion, where every
element except the anomaly is constant. Under J2 the node and the argument of perigee drift,
and the filters would have to be padded by the largest distance a point on the path can move
under that drift, which is roughly `r (|dOmega/dt| + |domega/dt|) T`. For a low Earth orbit
the nodal rate is of order 1e-6 rad/s, so over a one day window the required pad exceeds 500
km. A 5 km threshold padded to 500 km rejects almost nothing and the cascade stops earning
its cost.

The alternatives would be to re-run the cascade over short sub-windows, so the pad stays
small within each, or to work in mean elements and accept that the filter bounds apply to
mean rather than osculating geometry. Both are the right answer for an operational system.
Neither was implemented here, because the point being demonstrated is the conservativeness
argument, and a pure two-body propagator lets that argument be exact and lets the tests
assert it without a tolerance. The cost is recorded under known limitations.

### An analytic state transition matrix instead of central differences

The two-body state transition matrix has a closed form in terms of the Lagrange coefficients
and their partial derivatives. It would be faster and exact to rounding. It was not
implemented because the derivation is long, easy to get subtly wrong, and hard to test
independently, while the central-difference matrix is short and comes with a strong
independent check in the symplectic identity. The finite-difference matrix costs twelve
propagations per call, which at the catalogue sizes screened here is not the bottleneck: the
whole 240 object run takes 0.18 s.

### A diagonal RIC covariance instead of an element-space construction

The obvious way to build a synthetic covariance is to write down radial, in-track, and
cross-track standard deviations and put them on the diagonal of a RIC matrix. That was the
first implementation and it produced covariances that were wrong in a way that mattered.

A diagonal RIC covariance implies a semi-major axis uncertainty of the same order as the
radial position uncertainty and uncorrelated with everything else. A semi-major axis error
drives in-track position error linearly in time, at a rate of three halves of the mean motion
times the error, so over a one day window a few hundred metres of radial sigma produced
in-track sigmas above 50 km at the time of closest approach. Real orbit determination
covariances are strongly correlated between in-track position and radial velocity for exactly
this reason, and they do not grow like that.

The covariance is therefore built from uncorrelated uncertainties on the classical elements
and mapped into Cartesian coordinates with the Jacobian of the state with respect to the
elements. The semi-major axis uncertainty then appears explicitly and can be given the few
metres a tracked object actually has. The result is also positive semi-definite by
construction, since `J diag(sigma^2) J^T` is a sum of positive multiples of rank one outer
products. A test compares two objects that differ only in their semi-major axis uncertainty
and asserts that the looser one grows far more over a day, so the mechanism is pinned rather
than merely described.

### Sampling the encounter plane directly for the Monte Carlo check

Drawing from the two-dimensional plane distribution would be faster and would still validate
the quadrature. It was rejected because it validates nothing else. Drawing in three dimensions
and projecting exercises the covariance square root, the encounter plane basis, and the
projection, so a sign error or a transposed rotation anywhere in that chain shows up as a
disagreement rather than passing unnoticed.

### A three-dimensional or non-linear probability formulation

Methods exist for encounters where the linear relative motion assumption fails, including
numerical integration of the collision rate along the relative trajectory and the long-term
formulations in Chan's book. None is implemented. The two-dimensional formulation covers the
short duration, high relative velocity encounters that dominate low Earth orbit conjunction
assessment, and implementing a second regime properly would have doubled the work without
adding to what the project is demonstrating. The conditions under which the implemented
formulation fails are stated below rather than hidden.

## Known limitations

### The two-dimensional formulation assumes a short linear encounter

The probability computed here is valid when the relative motion through the encounter is
effectively a straight line at constant velocity, and when the combined covariance is
effectively constant over the same interval. Both assumptions hold when the relative speed is
high and the encounter lasts seconds, which is the usual case for two objects in low Earth
orbit crossing at a large angle.

They fail for slow encounters, where the two objects have similar velocities and remain in
proximity for a substantial fraction of an orbit. In that regime the relative trajectory
curves, the covariance evolves during the encounter, and the piercing point of a straight
line through a fixed plane is not the right question. They fail again for repeating
encounters, where two objects with nearly equal periods pass each other on many successive
revolutions; the per-pass probabilities are not independent, so summing them overstates the
total, and each pass may itself be slow. Nothing in this library detects either condition,
and both would need a different formulation rather than a tighter tolerance.

### The propagation model is two-body

There is no J2, no drag, no third body, and no solar radiation pressure. Real orbits drift
away from two-body predictions within hours, so the times of closest approach and the miss
distances reported here describe the model and not the sky. The synthetic catalogue is
generated under the same model, so the internal consistency the tests check is real, but no
number in this repository is a prediction about a real object. The section on rejected
alternatives records why the model was kept this simple and what it would take to change it.

### The covariance is synthetic and the two objects are treated as independent

The element uncertainties are drawn from plausible ranges rather than produced by an orbit
determination fit, and the correlations a real fit produces between elements are absent. The
combined relative covariance is the sum of the two individual covariances, which assumes the
two solutions are uncorrelated. Two objects tracked by the same sensor network share
atmospheric density model error and station bias, so their errors are correlated in practice,
and neglecting that biases the combined covariance.

### The hard body model is a sphere

Both objects are treated as spheres and the collision condition is a single combined radius.
That is the standard screening assumption and it is conservative for compact objects, but it
is poor for a large object with deployed structures, where the effective cross section depends
on the approach direction.

### Probability of collision is a decision input, not a decision

The number this library computes is one input to a manoeuvre decision. It does not account for
the cost of the manoeuvre, the propellant budget, the mission impact of moving, the risk
introduced by moving into a different part of the catalogue, the quality of the tracking
behind either covariance, or the possibility that the covariance itself is wrong. The action
thresholds in the analysis layer, 1e-4 for act and 1e-7 for monitor, follow common operational
practice, but they encode a policy about acceptable risk and are configurable for that reason.

The dilution behaviour makes the point sharply. A conjunction can report a small probability
because it is genuinely safe, or because the covariance is so large that the probability has
fallen off the far side of its peak. Those two situations call for opposite responses, and the
probability alone does not distinguish them. That is why the library also reports the maximum
probability over covariance scaling and the dilution factor: a small probability with a
dilution factor near one is evidence of safety, and a small probability with a large dilution
factor is evidence that the object needs more tracking.
