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


def _prepare_triplet_grid(grid, fallback_wavelengths=()):
    """Give absent model transitions zero emissivity on the full native grid.

    NaN from the raw reader means that the line is absent from that temperature's
    line list. XSPEC sums the listed transitions at each interpolation endpoint;
    an absent transition therefore contributes zero, independently of its peers.
    Wavelengths are transition identities, not temperature-dependent weights.
    """
    emissivities = np.array(grid.emissivities, dtype=float, copy=True)
    missing = np.isnan(emissivities)
    if (emissivities.shape != (len(grid.temperatures), 3)
            or np.any(np.isinf(emissivities)) or np.any(emissivities[~missing] < 0)):
        raise ValueError("invalid native triplet emissivities")
    fallback = tuple(fallback_wavelengths)
    wavelengths = np.empty(3)
    for column in range(3):
        listed = ~missing[:, column]
        values = grid.wavelengths[listed, column]
        if len(values):
            if (not np.isfinite(values).all() or np.any(values <= 0)
                    or not np.all(values == values[0])):
                raise ValueError("selected line wavelengths change across native temperatures")
            wavelengths[column] = values[0]
        elif len(fallback) == 3 and np.isfinite(fallback[column]) and fallback[column] > 0:
            wavelengths[column] = fallback[column]
        else:
            raise ValueError("absent transition requires a wavelength from matching atomic data")
    emissivities[missing] = 0.
    wavelengths = np.tile(wavelengths, (len(grid.temperatures), 1))
    for array in (emissivities, wavelengths):
        array.setflags(write=False)
    return _NativeLineGrid(grid.temperatures, emissivities, wavelengths, grid.lines)


def _integrate_triplet(grid, temperatures, weights):
    """Integrate all DEM bins using XSPEC's linear, endpoint-clamped table rule."""
    from ._uv_coverage import _validated_dem_weights

    temperatures, weights = _validated_dem_weights(temperatures, weights)
    active = weights > 0
    values = np.column_stack([
        np.interp(temperatures[active], grid.temperatures, grid.emissivities[:, column])
        for column in range(3)
    ])
    integral = weights[active] @ values
    if not np.isfinite(integral).all() or np.any(integral < 0):
        raise ValueError("invalid temperature-integrated triplet flux")
    return integral, grid.wavelengths[0].copy()


def _atomic_line_wavelengths(level_file, radiative_file, lines):
    """Resolve globally unlisted transitions from the matching LV/LA files."""
    from astropy.io import fits
    from .units import HC_KEV_ANGSTROM

    with fits.open(level_file, memmap=False) as hdus:
        energies = np.asarray(hdus[1].data["ENERGY"], dtype=float).copy()
    with fits.open(radiative_file, memmap=False) as hdus:
        rows = hdus[1].data
        result = []
        for line in lines:
            matched = rows[(rows["UPPER_LEV"] == line.upper)
                           & (rows["LOWER_LEV"] == line.lower)]
            wavelength = np.nan
            if len(matched) == 1:
                for name in ("WAVE_OBS", "WAVELEN", "WAVELENGTH", "LAMBDA"):
                    if name not in rows.dtype.names:
                        continue
                    candidate = float(matched[name][0])
                    if np.isfinite(candidate) and candidate > 0:
                        wavelength = candidate
                        break
            if not np.isfinite(wavelength) or wavelength <= 0:
                energy = energies[line.upper - 1] - energies[line.lower - 1]
                if not np.isfinite(energy) or energy <= 0:
                    raise ValueError("cannot resolve a missing triplet wavelength from atomic data")
                wavelength = HC_KEV_ANGSTROM * 1000. / energy  # LV energies are eV.
            result.append(wavelength)
    return tuple(result)


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

