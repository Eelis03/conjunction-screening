"""Property tests for the non-spherical hard body model.

Three claims carry the whole construction and each is checked directly rather
than through a downstream number.

* The combined body contains the Minkowski sum of the two bodies. That is a
  statement about support functions, so it is tested as one, over directions
  covering the sphere. A combined body that failed it would under-report the
  probability, which is the one direction of error that matters.
* The shadow of a sphere is the disc of the same radius from every direction, so
  attaching the machinery to a catalogue of spheres cannot move a number.
* The probability over an elliptical shadow lies between the probabilities over
  the inscribed and the circumscribed disc, because those regions are nested and
  the density is positive. That bracket needs no tolerance at all.

The quadrature tolerances are the ones the methods converge to, 1e-11 relative,
and the Monte Carlo comparison uses four binomial standard errors computed from
the estimate itself.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conjunction_screening.algorithm.probability import (
    AlfanoMethod,
    ChanMethod,
    FosterMethod,
    MonteCarloMethod,
    PateraMethod,
)
from conjunction_screening.model.encounter import (
    EncounterGeometry,
    planar_encounter,
    principal_axis_form,
)
from conjunction_screening.model.hardbody import (
    CrossSection,
    HardBody,
    combine_hard_bodies,
    projected_cross_section,
)
from conjunction_screening.pipeline.catalog import SyntheticCatalog
from conjunction_screening.pipeline.screening import ScreeningConfig, run_screening

_QUADRATURE_AGREEMENT = 1e-9
"""Allowed relative disagreement between two quadratures each converged to 1e-11."""


def _directions(count: int, seed: int) -> np.ndarray:
    """Return unit vectors spread over the sphere."""
    generator = np.random.default_rng(seed)
    raw = generator.standard_normal((count, 3))
    return np.asarray(raw / np.linalg.norm(raw, axis=1)[:, None], dtype=np.float64)


def _orthonormal_basis(seed: int) -> np.ndarray:
    """Return a 2 by 3 projection with orthonormal rows, in a random orientation."""
    generator = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(generator.standard_normal((3, 3)))
    return np.asarray(rotation[:, :2].T, dtype=np.float64)


def test_a_sphere_is_an_ellipsoid_with_equal_semi_axes() -> None:
    """The sphere is the special case of the general body, not a separate one."""
    body = HardBody.sphere(7.5)
    assert body.semi_axes_m == pytest.approx([7.5, 7.5, 7.5], rel=1e-15)
    assert body.is_sphere
    assert body.bounding_radius_m == pytest.approx(7.5, rel=1e-15)
    assert body.shape_matrix == pytest.approx(np.eye(3) * 7.5**2, rel=1e-15)


def test_an_ellipsoid_reports_its_semi_axes_in_decreasing_order() -> None:
    """The eigenvalues of the shape matrix are the squared semi-axes."""
    body = HardBody.ellipsoid((3.0, 12.0, 5.0))
    assert body.semi_axes_m == pytest.approx([12.0, 5.0, 3.0], rel=1e-12)
    assert not body.is_sphere
    assert body.bounding_radius_m == pytest.approx(12.0, rel=1e-12)


def test_a_rotated_ellipsoid_keeps_its_semi_axes() -> None:
    """Orientation moves the axes without changing their lengths."""
    angle = 0.6
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    body = HardBody.ellipsoid((20.0, 4.0, 4.0), orientation=rotation)
    assert body.semi_axes_m == pytest.approx([20.0, 4.0, 4.0], rel=1e-12)


def test_the_support_function_is_the_reach_of_the_body() -> None:
    """The support of an ellipsoid along a principal axis is that semi-axis."""
    body = HardBody.ellipsoid((9.0, 4.0, 2.0))
    assert body.support_m(np.array([1.0, 0.0, 0.0])) == pytest.approx(9.0, rel=1e-12)
    assert body.support_m(np.array([0.0, 1.0, 0.0])) == pytest.approx(4.0, rel=1e-12)
    assert body.support_m(np.array([0.0, 0.0, 5.0])) == pytest.approx(2.0, rel=1e-12)


def test_invalid_bodies_are_rejected() -> None:
    """A flat, a negative, or a sheared body is a modelling error, not a small one."""
    with pytest.raises(ValueError, match="positive definite"):
        HardBody(shape_matrix=np.diag([1.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="three positive lengths"):
        HardBody.ellipsoid((1.0, -2.0, 3.0))
    with pytest.raises(ValueError, match="orthogonal"):
        HardBody.ellipsoid((1.0, 2.0, 3.0), orientation=np.diag([2.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="radius_m must be positive"):
        HardBody.sphere(0.0)


def test_combining_two_spheres_recovers_the_combined_radius() -> None:
    """The convention every screening system uses has to come out unchanged.

    This is the property that lets the machinery be attached to a catalogue of
    spheres without moving a single pinned number.
    """
    for first_radius, second_radius in ((1.0, 1.0), (3.5, 0.75), (12.0, 0.1)):
        combined = combine_hard_bodies(
            HardBody.sphere(first_radius), HardBody.sphere(second_radius)
        )
        assert combined.is_sphere
        assert combined.semi_axes_m[0] == pytest.approx(first_radius + second_radius, rel=1e-12)


def test_the_combined_body_contains_the_minkowski_sum() -> None:
    """The combined body reaches at least as far as the two bodies together.

    One convex body contains another exactly when its support function dominates
    in every direction, and the support of a Minkowski sum is the sum of the
    supports. Under-reporting in any direction would make the probability an
    underestimate, so this is checked over directions covering the sphere rather
    than on the axes, where the construction is easiest to get right.
    """
    first = HardBody.ellipsoid((14.0, 3.0, 2.0))
    second = HardBody.ellipsoid((1.0, 6.0, 4.0), orientation=_orthogonal_rotation(0.9))
    combined = combine_hard_bodies(first, second)
    for direction in _directions(400, seed=91):
        required = first.support_m(direction) + second.support_m(direction)
        assert combined.support_m(direction) >= required * (1.0 - 1e-12)


def _orthogonal_rotation(angle_rad: float) -> np.ndarray:
    """Return a rotation about the third axis."""
    cosine, sine = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def test_the_combining_parameter_minimises_the_size_of_the_result() -> None:
    """No other member of the containing family is smaller in the trace.

    Every positive parameter gives a body that contains the sum, so the choice is
    free and has to be justified. Minimising the trace, which is the sum of the
    squared semi-axes, is the justification, and it is checked here against a
    sweep rather than asserted.
    """
    first = HardBody.ellipsoid((14.0, 3.0, 2.0))
    second = HardBody.ellipsoid((1.0, 6.0, 4.0))
    chosen = float(np.trace(combine_hard_bodies(first, second).shape_matrix))
    for parameter in np.geomspace(0.05, 20.0, 60):
        family = (1.0 + 1.0 / parameter) * first.shape_matrix + (
            1.0 + parameter
        ) * second.shape_matrix
        assert float(np.trace(family)) >= chosen * (1.0 - 1e-12)


def test_the_combined_body_is_an_outer_approximation_and_says_where_it_is_loose() -> None:
    """The price of using an ellipsoid for a sum that is not one, measured.

    Along the long axis of a boom-like body the approximation is tight, because
    that is close to the direction the chosen parameter is exact in. Across it
    the same body is overstated by four tenths of its own width, which makes the
    probability an overestimate. That is the safe direction to be wrong in, and
    it is recorded here so the size of the error is a measured number rather than
    an assurance.
    """
    boom = HardBody.ellipsoid((30.0, 3.0, 3.0))
    companion = HardBody.sphere(3.0)
    combined = combine_hard_bodies(boom, companion)

    along = np.array([1.0, 0.0, 0.0])
    across = np.array([0.0, 1.0, 0.0])
    exact_along = boom.support_m(along) + companion.support_m(along)
    exact_across = boom.support_m(across) + companion.support_m(across)
    assert combined.support_m(along) / exact_along == pytest.approx(1.012, abs=0.01)
    assert combined.support_m(across) / exact_across == pytest.approx(1.414, abs=0.05)


def test_a_sphere_casts_the_same_disc_from_every_direction() -> None:
    """The shadow of a sphere carries no direction information."""
    body = HardBody.sphere(4.25)
    for seed in range(8):
        section = projected_cross_section(body, _orthonormal_basis(seed))
        assert section.is_circular
        assert section.equivalent_radius_m == pytest.approx(4.25, rel=1e-12)
        assert section.semi_axes_m[0] == pytest.approx(4.25, rel=1e-12)


def test_the_shadow_of_an_elongated_body_depends_on_the_viewing_direction() -> None:
    """Looking along the long axis of a body hides it; looking across does not.

    This is the whole content of the limitation being closed. The same body seen
    from two directions presents cross sections that differ by a factor of five
    in area here.
    """
    body = HardBody.ellipsoid((25.0, 5.0, 5.0))
    along = projected_cross_section(body, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    across = projected_cross_section(body, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert along.is_circular
    assert along.semi_axes_m == pytest.approx((5.0, 5.0), rel=1e-12)
    assert not across.is_circular
    assert across.semi_axes_m == pytest.approx((25.0, 5.0), rel=1e-12)
    assert across.area_m2 == pytest.approx(5.0 * along.area_m2, rel=1e-12)


def test_the_cross_section_boundary_matches_the_polar_form_of_the_ellipse() -> None:
    """``radius_at`` is the ellipse in polar coordinates, and ``contains`` agrees with it."""
    major, minor, orientation = 18.0, 6.0, 0.4
    section = CrossSection.ellipse(major, minor, orientation)
    for angle in np.linspace(0.0, 2.0 * np.pi, 37):
        local = angle - orientation
        expected = (major * minor) / float(
            np.sqrt((minor * np.cos(local)) ** 2 + (major * np.sin(local)) ** 2)
        )
        radius = section.radius_at(float(angle))
        assert radius == pytest.approx(expected, rel=1e-12)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        assert bool(section.contains((0.999 * radius * direction)[None, :])[0])
        assert not bool(section.contains((1.001 * radius * direction)[None, :])[0])


def test_a_disc_cross_section_has_a_constant_radius() -> None:
    """The disc is the cross section whose polar boundary does not vary."""
    section = CrossSection.disc(9.0)
    radii = [section.radius_at(float(angle)) for angle in np.linspace(0.0, 2.0 * np.pi, 13)]
    assert radii == pytest.approx([9.0] * len(radii), rel=1e-14)
    assert section.area_m2 == pytest.approx(np.pi * 81.0, rel=1e-14)
    assert section.equivalent_radius_m == pytest.approx(9.0, rel=1e-14)


def test_cross_section_area_and_equivalent_radius_follow_the_semi_axes() -> None:
    """The equivalent radius is the radius of the disc of the same area."""
    section = CrossSection.ellipse(20.0, 5.0, 1.1)
    assert section.area_m2 == pytest.approx(np.pi * 100.0, rel=1e-12)
    assert section.equivalent_radius_m == pytest.approx(10.0, rel=1e-12)
    assert section.scaled(3.0).equivalent_radius_m == pytest.approx(30.0, rel=1e-12)


def test_rotating_a_cross_section_preserves_its_shape() -> None:
    """Expressing the same ellipse in another frame cannot change its semi-axes."""
    section = CrossSection.ellipse(20.0, 5.0)
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    rotated = section.rotated(rotation)
    assert rotated.semi_axes_m == pytest.approx(section.semi_axes_m, rel=1e-12)
    assert rotated.area_m2 == pytest.approx(section.area_m2, rel=1e-12)


def test_invalid_cross_sections_are_rejected() -> None:
    """A degenerate outline has no interior and no probability."""
    with pytest.raises(ValueError, match="positive definite"):
        CrossSection(matrix=np.diag([4.0, 0.0]))
    with pytest.raises(ValueError, match="positive"):
        CrossSection.ellipse(5.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        CrossSection.disc(-1.0)
    with pytest.raises(ValueError, match="shape"):
        CrossSection.disc(3.0).contains(np.zeros((4, 3)))


def test_the_general_quadrature_reproduces_the_disc_quadrature() -> None:
    """The two Foster paths integrate the same region in different orders.

    Without a cross section the radial limit is a constant and the angle is the
    inner variable, which is the published formulation. With one the angle is the
    outer variable and the radial limit is a function of it. Attaching an
    explicit disc runs the same encounter through the second path, and the two
    must agree to the tolerance each was asked for.
    """
    method = FosterMethod()
    for miss, sigma_x, sigma_y, radius in (
        (50.0, 100.0, 100.0, 10.0),
        (300.0, 1_000.0, 200.0, 12.0),
        (700.0, 300.0, 300.0, 8.0),
    ):
        encounter = planar_encounter(
            miss_distance_m=miss,
            sigma_x_m=sigma_x,
            sigma_y_m=sigma_y,
            hard_body_radius_m=radius,
        )
        disc_path = method.probability(encounter)
        general_path = method.probability(encounter.with_cross_section(CrossSection.disc(radius)))
        assert disc_path.converged
        assert general_path.converged
        assert general_path.value == pytest.approx(disc_path.value, rel=_QUADRATURE_AGREEMENT)


def test_an_elliptical_cross_section_is_bracketed_by_its_two_discs() -> None:
    """Nested regions give ordered probabilities, with no tolerance needed.

    The ellipse contains the disc of its minor semi-axis and is contained in the
    disc of its major semi-axis. The density is positive everywhere, so the three
    probabilities are strictly ordered. A quadrature that mishandled the variable
    radial limit would break this without needing to be far wrong.
    """
    method = FosterMethod()
    base = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    )
    inner = method.probability(base.with_cross_section(CrossSection.disc(5.0))).value
    ellipse = method.probability(base.with_cross_section(CrossSection.ellipse(20.0, 5.0))).value
    outer = method.probability(base.with_cross_section(CrossSection.disc(20.0))).value
    assert inner < ellipse < outer


def test_an_elongated_cross_section_collects_more_mass_pointing_at_the_density() -> None:
    """Orientation matters, and it matters in the direction the geometry predicts.

    The density peaks at the miss vector, which lies along the first in-plane
    axis. An outline extended along that axis reaches towards the peak, so it
    collects more mass than the same outline turned across it. Area alone cannot
    order these two: they have the same area.
    """
    method = FosterMethod()
    base = planar_encounter(
        miss_distance_m=400.0, sigma_x_m=300.0, sigma_y_m=300.0, hard_body_radius_m=10.0
    )
    towards = method.probability(base.with_cross_section(CrossSection.ellipse(40.0, 2.5))).value
    across = method.probability(
        base.with_cross_section(CrossSection.ellipse(40.0, 2.5, 0.5 * np.pi))
    ).value
    assert towards > across


def test_patera_agrees_with_the_quadrature_over_an_ellipse() -> None:
    """The analytic cross check that survives the shape change.

    Alfano and Chan use the circle before the density enters, so neither extends
    to an elliptical outline. Patera's contour integral never uses it: the region
    appears only as the curve bounding it, so an ellipse costs a different curve
    and not a different formulation. That restores a second analytic evaluation
    for the shaped body, at the same tolerance the disc case is checked to rather
    than at the width of a sampling error.
    """
    method = FosterMethod()
    contour = PateraMethod()
    base = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    )
    for major, minor, orientation in ((20.0, 5.0, 0.3), (40.0, 2.5, 0.0), (12.0, 8.0, 1.2)):
        encounter = base.with_cross_section(CrossSection.ellipse(major, minor, orientation))
        area = method.probability(encounter)
        boundary = contour.probability(encounter)
        assert area.converged
        assert boundary.converged
        assert boundary.value == pytest.approx(area.value, rel=_QUADRATURE_AGREEMENT)


def test_patera_reads_an_elliptical_outline_as_the_ellipse_and_not_its_area() -> None:
    """Two outlines of equal area pointed differently give different answers.

    An equal-area disc substitution, which is what Chan would have to make, cannot
    tell these two apart. The contour method separates them because the curve it
    integrates around is the outline itself, and the ordering it produces is the
    one the geometry requires: the outline reaching towards the density peak
    collects more mass.
    """
    method = PateraMethod()
    base = planar_encounter(
        miss_distance_m=400.0, sigma_x_m=300.0, sigma_y_m=300.0, hard_body_radius_m=10.0
    )
    towards = method.probability(base.with_cross_section(CrossSection.ellipse(40.0, 2.5)))
    across = method.probability(
        base.with_cross_section(CrossSection.ellipse(40.0, 2.5, 0.5 * np.pi))
    )
    disc = method.probability(base.with_cross_section(CrossSection.disc(10.0)))
    assert towards.value > disc.value > across.value


def test_monte_carlo_agrees_with_the_quadrature_over_an_ellipse() -> None:
    """A check of the elliptical path that goes through the plane construction too.

    Foster and Patera cross validate each other on the outline analytically, but
    both start from the same principal axis reduction of an encounter that is
    already planar. Sampling in three dimensions and projecting covers the
    covariance square root and the projection as well. Tolerance: four binomial
    standard errors computed from the estimate itself.
    """
    encounter = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    ).with_cross_section(CrossSection.ellipse(20.0, 5.0, 0.3))
    analytic = FosterMethod().probability(encounter)
    sampled = MonteCarloMethod(samples=2_000_000, seed=6151).probability(encounter)
    assert analytic.converged
    assert sampled.converged
    assert abs(sampled.value - analytic.value) <= 4.0 * sampled.error_estimate


def test_the_series_methods_refuse_a_non_circular_cross_section() -> None:
    """Alfano and Chan say so rather than quietly answering a different question."""
    encounter = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    ).with_cross_section(CrossSection.ellipse(20.0, 5.0))
    with pytest.raises(ValueError, match="circular hard body"):
        AlfanoMethod().probability(encounter)
    with pytest.raises(ValueError, match="circular hard body"):
        ChanMethod().probability(encounter)


def test_the_series_methods_accept_a_circular_cross_section() -> None:
    """A disc is a disc however it was arrived at."""
    encounter = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    )
    attached = encounter.with_cross_section(CrossSection.disc(10.0))
    assert AlfanoMethod().probability(attached).value == pytest.approx(
        AlfanoMethod().probability(encounter).value, rel=1e-14
    )
    assert ChanMethod().probability(attached).value == pytest.approx(
        ChanMethod().probability(encounter).value, rel=1e-14
    )


def test_the_reported_radius_and_the_cross_section_cannot_disagree() -> None:
    """One encounter carries one hard body size, whichever way it is read."""
    base = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    )
    attached = base.with_cross_section(CrossSection.ellipse(20.0, 5.0))
    assert attached.hard_body_radius_m == pytest.approx(10.0, rel=1e-12)

    with pytest.raises(ValueError, match="area-equivalent radius"):
        EncounterGeometry(
            tca_s=base.tca_s,
            relative_position_m=base.relative_position_m,
            relative_velocity_m_s=base.relative_velocity_m_s,
            relative_covariance=base.relative_covariance,
            relative_covariance_ric=base.relative_covariance_ric,
            basis=base.basis,
            miss_vector_m=base.miss_vector_m,
            plane_covariance=base.plane_covariance,
            hard_body_radius_m=3.0,
            cross_section=CrossSection.ellipse(20.0, 5.0),
        )


def test_resizing_an_encounter_keeps_the_shape_of_its_cross_section() -> None:
    """Changing how big the body is must not change what shape it is."""
    attached = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    ).with_cross_section(CrossSection.ellipse(20.0, 5.0))
    resized = attached.with_hard_body_radius(20.0)
    assert resized.cross_section is not None
    major, minor = resized.cross_section.semi_axes_m
    assert major / minor == pytest.approx(4.0, rel=1e-12)
    assert resized.hard_body_radius_m == pytest.approx(20.0, rel=1e-12)
    assert resized.cross_section.equivalent_radius_m == pytest.approx(20.0, rel=1e-12)


def test_scaling_the_covariance_carries_the_cross_section() -> None:
    """The dilution study scales covariances, and the body must come along unchanged."""
    attached = planar_encounter(
        miss_distance_m=150.0, sigma_x_m=300.0, sigma_y_m=200.0, hard_body_radius_m=10.0
    ).with_cross_section(CrossSection.ellipse(20.0, 5.0))
    scaled = attached.with_scaled_covariance(4.0)
    assert scaled.cross_section is not None
    assert scaled.cross_section.semi_axes_m == pytest.approx((20.0, 5.0), rel=1e-12)


def test_the_principal_form_rotates_the_cross_section_with_the_covariance() -> None:
    """The outline has to be read in the same axes the density is diagonal in.

    The reduction to principal axes rotates the frame. An outline left behind in
    the old frame would be silently mis-oriented, which for an elongated body
    changes the answer.
    """
    encounter = planar_encounter(
        miss_distance_m=150.0,
        sigma_x_m=900.0,
        sigma_y_m=200.0,
        hard_body_radius_m=10.0,
        orientation_rad=0.6,
    ).with_cross_section(CrossSection.ellipse(20.0, 5.0, 0.6))
    form = principal_axis_form(encounter)
    assert form.cross_section is not None
    assert form.cross_section.semi_axes_m == pytest.approx((20.0, 5.0), rel=1e-9)
    # The covariance principal axis is at 0.6 rad and so is the outline, so in the
    # principal frame the outline is aligned with the first axis and the matrix is
    # diagonal there.
    assert float(abs(form.cross_section.matrix[0, 1])) < 1e-9 * 20.0**2


def test_a_catalogue_of_spheres_attaches_no_cross_section(
    mixed_catalog: SyntheticCatalog,
) -> None:
    """The disc path stays the path a spherical catalogue takes."""
    report = run_screening(mixed_catalog, ScreeningConfig.for_threshold(5_000.0))
    assert report.events
    for event in report.events:
        assert event.encounter.cross_section is None


def test_a_shaped_primary_produces_a_direction_dependent_cross_section(
    mixed_catalog: SyntheticCatalog,
) -> None:
    """End to end, an elongated primary gives every event its own outline.

    The bodies are combined once and the shadow is cast along each event's own
    relative velocity, so the events see different outlines of the same object.
    That difference is the limitation being closed, and it is visible in the
    screening report rather than only in the model layer.
    """
    shaped = replace(mixed_catalog.primary, shape=HardBody.ellipsoid((30.0, 3.0, 3.0)))
    catalog = replace(mixed_catalog, primary=shaped)
    report = run_screening(catalog, ScreeningConfig.for_threshold(5_000.0))
    assert report.events

    secondary_radii = {item.object_id: item.radius_m for item in mixed_catalog.secondaries}
    majors = []
    for event in report.events:
        section = event.encounter.cross_section
        assert section is not None
        major, minor = section.semi_axes_m
        combined_short = 3.0 + secondary_radii[event.object_id]
        combined_long = 30.0 + secondary_radii[event.object_id]
        # The shadow contains the shadow of the inscribed sphere of the combined
        # body and can never be longer than the combined body itself.
        assert minor >= combined_short
        assert major <= 1.1 * combined_long
        assert event.encounter.hard_body_radius_m == pytest.approx(
            section.equivalent_radius_m, rel=1e-12
        )
        majors.append(major / combined_long)
    assert max(majors) > min(majors) * 1.1


def test_a_shaped_primary_changes_the_probability(mixed_catalog: SyntheticCatalog) -> None:
    """A body with thirty times the cross section does not report the same risk.

    The comparison is against the same catalogue with the primary left spherical,
    so the only thing that differs is the hard body geometry.
    """
    spherical = run_screening(mixed_catalog, ScreeningConfig.for_threshold(5_000.0))
    shaped = replace(mixed_catalog.primary, shape=HardBody.ellipsoid((30.0, 3.0, 3.0)))
    elongated = run_screening(
        replace(mixed_catalog, primary=shaped), ScreeningConfig.for_threshold(5_000.0)
    )

    by_id = {event.object_id: event for event in spherical.events}
    compared = 0
    for event in elongated.events:
        reference = by_id.get(event.object_id)
        if reference is None or reference.probability.value <= 0.0:
            continue
        assert event.encounter.hard_body_radius_m > reference.encounter.hard_body_radius_m
        assert event.probability.value > reference.probability.value
        compared += 1
    assert compared > 0
