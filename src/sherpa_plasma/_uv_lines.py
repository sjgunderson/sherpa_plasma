"""Read selected native-grid APEC line data without interpolating it."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .plasma import LineID


@dataclass(frozen=True)
class _NativeLineGrid:
    temperatures: np.ndarray
    emissivities: np.ndarray
    wavelengths: np.ndarray
    lines: tuple


def _interpolate_native_lines(grid, temperature):
    """Interpolate complete native lines linearly in temperature.

    Only the explicitly present transitions are used; a missing line on
    either endpoint is an error. This does not apply abundance or XSPEC norm.
    """
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    temperatures = grid.temperatures
    index = int(np.searchsorted(temperatures, temperature))
    if index < len(temperatures) and temperature == temperatures[index]:
        emissivities = grid.emissivities[index]
        wavelengths = grid.wavelengths[index]
    else:
        if index == 0 or index == len(temperatures):
            raise ValueError("temperature is outside the native APEC grid")
        lo, hi = index - 1, index
        emissivity_pair = grid.emissivities[[lo, hi]]
        wavelength_pair = grid.wavelengths[[lo, hi]]
        if not np.isfinite(emissivity_pair).all() or not np.isfinite(wavelength_pair).all():
            raise ValueError("selected triplet is incomplete at an enclosing temperature")
        if not np.array_equal(wavelength_pair[0], wavelength_pair[1]):
            raise ValueError("selected line wavelengths change across interpolation endpoints")
        weight = (temperature - temperatures[lo]) / (temperatures[hi] - temperatures[lo])
        emissivities = emissivity_pair[0] * (1 - weight) + emissivity_pair[1] * weight
        wavelengths = wavelength_pair[0]
    if not np.isfinite(emissivities).all() or not np.isfinite(wavelengths).all():
        raise ValueError("selected triplet is incomplete at this temperature")
    return emissivities.copy(), wavelengths.copy()


def _read_native_lines(path, lines):
    """Extract native emissivities; NaN denotes an absent transition.

    This is not an XSPEC-compatible interpolation or flux adapter. The caller
    must validate transition identities and handle missing entries explicitly.
    """
    from astropy.io import fits

    lines = tuple(lines)
    if not lines or any(not isinstance(line, LineID) for line in lines):
        raise ValueError("lines must contain at least one LineID")
    if len(set(lines)) != len(lines):
        raise ValueError("line identities must be distinct")

    with fits.open(Path(path), memmap=True) as hdus:
        temperatures = np.array(hdus[1].data["kT"], dtype=float, copy=True)
        if (temperatures.ndim != 1 or temperatures.size == 0
                or not np.isfinite(temperatures).all()
                or np.any(temperatures <= 0)
                or np.any(np.diff(temperatures) <= 0)):
            raise ValueError("APEC temperature grid must be finite and strictly increasing")
        if len(hdus) != temperatures.size + 2:
            raise ValueError("APEC temperature and line extension counts do not match")
        emissivities = np.full((len(temperatures), len(lines)), np.nan)
        wavelengths = np.full_like(emissivities, np.nan)
        for index, hdu in enumerate(hdus[2:]):
            rows = hdu.data
            for column, line in enumerate(lines):
                matches = rows[line.mask(rows)]
                if len(matches) > 1:
                    raise ValueError(f"duplicate APEC line {line} at grid index {index}")
                if not len(matches):
                    continue
                emissivity = float(matches["Epsilon"][0])
                wavelength = float(matches["Lambda"][0])
                if (not np.isfinite(emissivity) or emissivity < 0
                        or not np.isfinite(wavelength) or wavelength <= 0):
                    raise ValueError(f"invalid APEC line {line} at grid index {index}")
                emissivities[index, column] = emissivity
                wavelengths[index, column] = wavelength

    for array in (temperatures, emissivities, wavelengths):
        array.setflags(write=False)
    return _NativeLineGrid(temperatures, emissivities, wavelengths, lines)
