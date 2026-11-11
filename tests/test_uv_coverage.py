"""Full-DEM integration without Sherpa, XSPEC, or FITS dependencies."""

import numpy as np
import pytest
import sys
from types import ModuleType, SimpleNamespace

from sherpa_plasma._uv_coverage import _validated_dem_weights
from sherpa_plasma._uv_lines import _NativeLineGrid, _integrate_triplet, _prepare_triplet_grid
from sherpa_plasma._uv_lines import _atomic_line_wavelengths
from sherpa_plasma.plasma import LineID
from sherpa_plasma.units import HC_KEV_ANGSTROM


def sparse_grid():
    temperatures = np.array([.1, .2, 1., 2., 3.])
    emissivities = np.array([
        [2., np.nan, np.nan], [4., 2., np.nan], [8., 4., 2.],
        [3., np.nan, 1.], [1., np.nan, np.nan],
    ])
    wavelengths = np.tile([7., 6.9, 6.8], (5, 1))
    wavelengths[np.isnan(emissivities)] = np.nan
    return _NativeLineGrid(temperatures, emissivities, wavelengths, ())


def test_all_temperatures_contribute_even_without_complete_triplets():
    raw = sparse_grid()
    before = raw.emissivities.copy()
    grid = _prepare_triplet_grid(raw)
    actual, wavelengths = _integrate_triplet(grid, raw.temperatures, np.ones(5))
    np.testing.assert_array_equal(actual, [18., 6., 3.])
    np.testing.assert_array_equal(wavelengths, [7., 6.9, 6.8])
    np.testing.assert_array_equal(raw.emissivities, before)
    assert not grid.emissivities.flags.writeable


def test_each_line_interpolates_independently_through_missing_endpoints():
    grid = _prepare_triplet_grid(sparse_grid())
    actual, _ = _integrate_triplet(grid, [.15], [2.])
    np.testing.assert_allclose(actual, [6., 2., 0.])


def test_broad_dem_preserves_absolute_weights_and_clamps_native_endpoints():
    grid = _prepare_triplet_grid(sparse_grid())
    actual, _ = _integrate_triplet(grid, [.001, .15, 1., 100.], [3., 2., 4., 5.])
    np.testing.assert_allclose(actual, [49., 18., 8.])


def test_no_overlap_with_former_complete_range_still_has_forbidden_flux():
    grid = _prepare_triplet_grid(sparse_grid())
    actual, _ = _integrate_triplet(grid, [.1, 3.], [2., 3.])
    np.testing.assert_array_equal(actual, [7., 0., 0.])


def test_zero_weight_bins_do_not_contribute():
    grid = _prepare_triplet_grid(sparse_grid())
    actual, _ = _integrate_triplet(grid, [.1, .2, 1., 2., 3.], [0., 0., 1., 0., 0.])
    np.testing.assert_array_equal(actual, [8., 4., 2.])


def test_globally_unlisted_branch_uses_atomic_wavelength_and_zero_emissivity():
    raw = sparse_grid()
    raw.emissivities[:, 2] = np.nan
    raw.wavelengths[:, 2] = np.nan
    grid = _prepare_triplet_grid(raw, [7., 6.9, 6.8])
    actual, wavelengths = _integrate_triplet(grid, [1.], [1.])
    np.testing.assert_array_equal(actual, [8., 4., 0.])
    np.testing.assert_array_equal(wavelengths, [7., 6.9, 6.8])


def test_entirely_absent_triplet_has_zero_integral():
    raw = sparse_grid()
    raw.emissivities[:] = np.nan
    raw.wavelengths[:] = np.nan
    grid = _prepare_triplet_grid(raw, [7., 6.9, 6.8])
    np.testing.assert_array_equal(_integrate_triplet(grid, [.001, 100.], [1., 1.])[0], 0.)


def test_missing_atomic_wavelength_is_an_identity_error():
    raw = sparse_grid()
    raw.emissivities[:, 2] = np.nan
    with pytest.raises(ValueError, match="requires a wavelength"):
        _prepare_triplet_grid(raw)


def test_present_wavelength_changes_remain_an_error():
    raw = sparse_grid()
    raw.wavelengths[0, 0] = 6.
    with pytest.raises(ValueError, match="wavelengths change"):
        _prepare_triplet_grid(raw)


@pytest.mark.parametrize("valid_energy", [True, False])
def test_atomic_wavelength_fallback_order_with_controlled_fits(monkeypatch, valid_energy):
    levels = np.array([(0.,), (1000.,), (1050.,), (1100.,),
                       (1200. if valid_energy else 0.,)], dtype=[("ENERGY", float)])
    radiative = np.array([(2, 1, 12.401, 12.4), (4, 1, 0., 11.7), (5, 1, 0., 0.)],
                         dtype=[("UPPER_LEV", int), ("LOWER_LEV", int),
                                ("WAVE_OBS", float), ("WAVELEN", float)])
    class HDUs(list):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    fits = ModuleType("astropy.io.fits")
    fits.open = lambda path, **kwargs: HDUs([
        None, SimpleNamespace(data=levels if path == "levels" else radiative)])
    astropy, io = ModuleType("astropy"), ModuleType("astropy.io")
    astropy.io, io.fits = io, fits
    for name, module in (("astropy", astropy), ("astropy.io", io), ("astropy.io.fits", fits)):
        monkeypatch.setitem(sys.modules, name, module)
    lines = [LineID(14, 13, upper, 1) for upper in (2, 4, 5)]
    if valid_energy:
        np.testing.assert_allclose(_atomic_line_wavelengths("levels", "rates", lines),
                                   [12.401, 11.7, HC_KEV_ANGSTROM * 1000. / 1200.])
    else:
        with pytest.raises(ValueError, match="cannot resolve a missing triplet wavelength"):
            _atomic_line_wavelengths("levels", "rates", lines)


@pytest.mark.parametrize("temperatures,weights", [
    ([1.], [-1.]), ([1.], [0.]), ([1.], [float("nan")]),
    ([float("nan")], [1.]), ([0.], [1.]), ([1., 2.], [1.]), ([], []),
])
def test_invalid_dem_is_rejected(temperatures, weights):
    with pytest.raises(ValueError, match="invalid DEM temperature weights"):
        _validated_dem_weights(temperatures, weights)
