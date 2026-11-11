"""Exercise the coverage error budget without requiring Sherpa or XSPEC."""

import numpy as np
import pytest

from sherpa_plasma._uv_coverage import (
    _COVERAGE_TOLERANCE, _coverage_bound, _dem_active_weights,
)
from sherpa_plasma._uv_lines import _NativeLineGrid


@pytest.fixture
def triplet():
    temperatures = np.array([.1, .2, 1., 2., 3.])
    emissivities = np.array([
        [1.e-9, 1.e-9, np.nan],
        [1.e-8, 2.e-8, 1.e-8],
        [4., 2., 1.],
        [2.e-8, 1.e-8, 1.e-8],
        [1.e-9, 0., 0.],
    ])
    wavelengths = np.tile([7., 6.9, 6.8], (5, 1))
    wavelengths[~np.isfinite(emissivities)] = np.nan
    grid = _NativeLineGrid(temperatures, emissivities, wavelengths, ())
    record = {"temperature_min_keV": .2, "temperature_max_keV": 2.}
    return record, grid


def test_negligible_positive_tails_are_selected_out_at_both_edges(triplet):
    record, grid = triplet
    temperatures = np.array([.1, .15, 1., 2.5, 3.])
    weights = np.ones(5)
    before = grid.emissivities.copy()
    mask = _dem_active_weights(temperatures, weights, record, grid, "Si")
    np.testing.assert_array_equal(mask, [False, False, True, False, False])
    np.testing.assert_array_equal(weights, np.ones(5))
    np.testing.assert_array_equal(grid.emissivities, before)
    covered = (temperatures >= .2) & (temperatures <= 2.)
    bound = _coverage_bound(temperatures, weights, record, grid, weights > 0, covered)
    assert 0 < bound < _COVERAGE_TOLERANCE


def test_zero_weight_tails_do_not_need_complete_line_data(triplet):
    record, grid = triplet
    mask = _dem_active_weights([.001, 1., 100.], [0., 1., 0.], record, grid, "Si")
    np.testing.assert_array_equal(mask, [False, True, False])


def test_closed_endpoints_and_python_list_inputs(triplet):
    record, grid = triplet
    mask = _dem_active_weights([.2, 2.], [1., 1.], record, grid, "Si")
    np.testing.assert_array_equal(mask, [True, True])


@pytest.mark.parametrize("temperature", [.1, 3.])
def test_single_temperature_no_overlap_is_an_explicit_value_error(triplet, temperature):
    record, grid = triplet
    with pytest.raises(ValueError, match="Si temperature is outside.*no overlap"):
        _dem_active_weights([temperature], [1.], record, grid, "Si")


@pytest.mark.parametrize("fraction", [1.e-4 * .999, 1.e-4 * 1.001])
@pytest.mark.parametrize("temperature", [.1, 3.])
def test_fraction_just_below_or_above_tolerance(fraction, temperature):
    record = {"temperature_min_keV": .2, "temperature_max_keV": 2.}
    grid = _NativeLineGrid(
        np.array([.2, 1., 2.]), np.ones((3, 3)), np.ones((3, 3)), ())
    weights = [fraction / (1 - fraction), 1.]
    if fraction < _COVERAGE_TOLERANCE:
        np.testing.assert_array_equal(
            _dem_active_weights([temperature, 1.], weights, record, grid, "Si"),
            [False, True])
    else:
        with pytest.raises(ValueError) as error:
            _dem_active_weights([temperature, 1.], weights, record, grid, "Si")
        message = str(error.value)
        assert "1 of 2 active DEM bins" in message
        assert f"{temperature:.5f}-{temperature:.5f} keV" in message
        assert "0.20000-2.00000 keV" in message
        assert "bound 0.0001001 exceeds tolerance 0.0001" in message


def test_bound_uses_worst_individual_line():
    record = {"temperature_min_keV": .2, "temperature_max_keV": 2.}
    grid = _NativeLineGrid(
        np.array([.2, 1., 2.]),
        np.array([[1., 1., 1.], [1.e6, 1.e6, 1.], [1., 1., 1.]]),
        np.ones((3, 3)), ())
    with pytest.raises(ValueError, match="bound.*exceeds tolerance"):
        _dem_active_weights([.1, 1.], [.001, 1.], record, grid, "Si")


def test_identically_zero_line_does_not_exhaust_error_budget():
    record = {"temperature_min_keV": .2, "temperature_max_keV": 2.}
    grid = _NativeLineGrid(
        np.array([.2, 1., 2.]),
        np.array([[1.e-8, 0., 1.e-8], [1., 0., 1.], [1.e-8, 0., 1.e-8]]),
        np.ones((3, 3)), ())
    np.testing.assert_array_equal(
        _dem_active_weights([.1, 1.], [1., 1.], record, grid, "Si"), [False, True])


def test_unverified_native_tail_is_not_treated_as_bounded(triplet):
    record, grid = triplet
    grid.emissivities[0, 0] = 100.
    with pytest.raises(ValueError, match="cannot bound uncovered triplet emission"):
        _dem_active_weights([.1, 1.], [1., 1.], record, grid, "Si")


def test_strict_and_custom_tolerances(triplet):
    record, grid = triplet
    with pytest.raises(ValueError, match="exceeds tolerance 0"):
        _dem_active_weights([.1, 1.], [1., 1.], record, grid, tolerance=0.)
    with pytest.raises(ValueError, match="exceeds tolerance"):
        _dem_active_weights([.1, 1.], [1.e5, 1.], record, grid)
    np.testing.assert_array_equal(
        _dem_active_weights([.1, 1.], [1.e5, 1.], record, grid, tolerance=.01),
        [False, True])


@pytest.mark.parametrize("tolerance", [-1., 1., float("inf"), float("nan")])
def test_invalid_tolerance_is_rejected(triplet, tolerance):
    record, grid = triplet
    with pytest.raises(ValueError, match="coverage tolerance"):
        _dem_active_weights([1.], [1.], record, grid, tolerance=tolerance)


@pytest.mark.parametrize("temperatures,weights", [
    ([1.], [-1.]), ([1.], [0.]), ([1.], [float("nan")]),
    ([float("nan")], [1.]), ([0.], [1.]), ([1., 2.], [1.]), ([], []),
])
def test_invalid_dem_is_rejected(triplet, temperatures, weights):
    record, grid = triplet
    with pytest.raises(ValueError, match="invalid DEM temperature weights"):
        _dem_active_weights(temperatures, weights, record, grid)


def test_bound_is_invariant_under_large_weight_scaling(triplet):
    record, grid = triplet
    temperatures = np.array([.1, 1., 3.])
    active = np.ones(3, dtype=bool)
    covered = np.array([False, True, False])
    small = _coverage_bound(temperatures, np.ones(3), record, grid, active, covered)
    large = _coverage_bound(temperatures, np.full(3, 1.e308), record, grid, active, covered)
    assert large == pytest.approx(small)
