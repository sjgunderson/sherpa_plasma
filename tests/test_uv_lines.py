"""Native-grid extraction tests independent of XSPEC interpolation."""

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from sherpa_plasma._uv_lines import _interpolate_native_lines, _read_native_lines
from sherpa_plasma.plasma import LineID


def _write_apec(path, rows_per_temperature, temperatures=(0.2, 0.3)):
    params = fits.BinTableHDU.from_columns([
        fits.Column(name="kT", format="E", array=temperatures),
    ])
    hdus = [fits.PrimaryHDU(), params]
    for rows in rows_per_temperature:
        values = list(zip(*rows)) if rows else [()] * 6
        columns = [
            fits.Column(name=name, format=fmt, array=values[idx])
            for idx, (name, fmt) in enumerate((
                ("Element", "J"), ("Ion", "J"), ("UpperLev", "J"),
                ("LowerLev", "J"), ("Epsilon", "E"), ("Lambda", "E"),
            ))
        ]
        hdus.append(fits.BinTableHDU.from_columns(columns))
    fits.HDUList(hdus).writeto(path)


def test_extracts_only_exact_lines_and_retains_missing_entries(tmp_path):
    path = tmp_path / "line.fits"
    f = LineID(11, 10, 2, 1)
    i = LineID(11, 10, 4, 1)
    _write_apec(path, [
        [(11, 10, 2, 1, 2.0, 11.19), (11, 9, 2, 1, 100., 11.19)],
        [(11, 10, 2, 1, 3.0, 11.19), (11, 10, 4, 1, 1.0, 11.08)],
    ])

    grid = _read_native_lines(path, [f, i])
    np.testing.assert_allclose(grid.temperatures, [0.2, 0.3])
    np.testing.assert_allclose(grid.emissivities, [[2, np.nan], [3, 1]])
    np.testing.assert_allclose(grid.wavelengths[1], [11.19, 11.08], rtol=1.e-6)
    assert grid.lines == (f, i)
    assert all(not a.flags.writeable for a in (
        grid.temperatures, grid.emissivities, grid.wavelengths
    ))


def test_rejects_duplicate_transition_entries(tmp_path):
    path = tmp_path / "line.fits"
    row = (11, 10, 2, 1, 2., 11.19)
    _write_apec(path, [[row, row], []])
    with pytest.raises(ValueError, match="duplicate APEC line"):
        _read_native_lines(path, [LineID(11, 10, 2, 1)])


def test_rejects_invalid_grid_and_lines(tmp_path):
    path = tmp_path / "line.fits"
    _write_apec(path, [[], []], temperatures=(0.3, 0.2))
    with pytest.raises(ValueError, match="strictly increasing"):
        _read_native_lines(path, [LineID(11, 10, 2, 1)])
    with pytest.raises(ValueError, match="distinct"):
        _read_native_lines(path, [LineID(11, 10, 2, 1)] * 2)


def test_linear_interpolation_and_incomplete_endpoints(tmp_path):
    path = tmp_path / "line.fits"
    f = LineID(11, 10, 2, 1)
    i = LineID(11, 10, 4, 1)
    _write_apec(path, [
        [(11, 10, 2, 1, 2., 11.19)],
        [(11, 10, 2, 1, 3., 11.19), (11, 10, 4, 1, 1., 11.08)],
    ])
    grid = _read_native_lines(path, [f])
    midpoint = (grid.temperatures[0] + grid.temperatures[1]) / 2
    emissivity, wavelength = _interpolate_native_lines(grid, midpoint)
    np.testing.assert_allclose(emissivity, [2.5])
    np.testing.assert_allclose(wavelength, [11.19], rtol=1.e-6)
    np.testing.assert_allclose(
        _interpolate_native_lines(grid, grid.temperatures[1])[0], [3.]
    )
    with pytest.raises(ValueError, match="outside"):
        _interpolate_native_lines(grid, .1)
    incomplete = _read_native_lines(path, [f, i])
    with pytest.raises(ValueError, match="incomplete"):
        _interpolate_native_lines(incomplete, midpoint)
    with pytest.raises(ValueError, match="incomplete"):
        _interpolate_native_lines(incomplete, grid.temperatures[0])


def test_rejects_line_wavelength_changes(tmp_path):
    path = tmp_path / "line.fits"
    _write_apec(path, [
        [(11, 10, 2, 1, 2., 11.19)],
        [(11, 10, 2, 1, 3., 11.18)],
    ])
    grid = _read_native_lines(path, [LineID(11, 10, 2, 1)])
    with pytest.raises(ValueError, match="wavelengths change"):
        _interpolate_native_lines(grid, .25)
