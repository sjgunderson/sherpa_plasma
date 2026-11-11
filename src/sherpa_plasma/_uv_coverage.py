"""Temperature-coverage checks independent of Sherpa and XSPEC."""

import numpy as np

from ._uv_lines import _interpolate_native_lines


def _validated_dem_weights(temperatures, weights):
    temperatures = np.asarray(temperatures, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (temperatures.ndim != 1 or temperatures.size == 0
            or temperatures.shape != weights.shape
            or not np.isfinite(temperatures).all()
            or np.any(temperatures <= 0) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("invalid DEM temperature weights")
    return temperatures, weights


#: Maximum upper bound on the discarded fraction of any individual line.
_COVERAGE_TOLERANCE = 1.e-4


def _coverage_bound(temperatures, weights, record, grid, active, covered):
    """Bound the triplet integral discarded outside complete coverage.

    Returns the largest per-line fraction of the integral that dropping the
    uncovered bins can discard. Use the nearest coverage edge as an emissivity
    ceiling, checking it against every finite native tail value before relying
    on it. Missing transitions are never interpolated or filled into the DEM.
    """
    low = record["temperature_min_keV"]
    high = record["temperature_max_keV"]
    edges = []
    for temperature in (low, high):
        emissivities, _ = _interpolate_native_lines(grid, temperature)
        edges.append(np.asarray(emissivities, dtype=float))
    for outside, native_tail, edge in (
            (active & (temperatures < low), grid.temperatures < low, edges[0]),
            (active & (temperatures > high), grid.temperatures > high, edges[1])):
        if np.any(outside) and np.any(grid.emissivities[native_tail] > edge):
            raise ValueError(
                "cannot bound uncovered triplet emission: native tail emissivity "
                "exceeds its complete-coverage edge")
    # Common scaling leaves the fraction unchanged and avoids large DEM weights
    # overflowing an otherwise well-defined ratio.
    weights = weights / weights[active].max()
    retained = np.zeros(3)
    for temperature, weight in zip(temperatures[active & covered],
                                   weights[active & covered]):
        emissivities, _ = _interpolate_native_lines(grid, temperature)
        retained += weight * np.asarray(emissivities, dtype=float)
    discarded = np.zeros(3)
    for temperature, weight in zip(temperatures[active & ~covered],
                                   weights[active & ~covered]):
        discarded += weight * (edges[0] if temperature < low else edges[1])
    total = retained + discarded
    fraction = np.divide(discarded, total, out=np.zeros(3), where=total > 0)
    return float(np.max(fraction))


def _dem_active_weights(temperatures, weights, record, grid, symbol=None, *,
                        tolerance=_COVERAGE_TOLERANCE):
    """Select the DEM bins usable for the triplet integral.

    Bins outside the complete-triplet range are dropped when the emission they
    discard is provably negligible, and reported as an error when it is not.
    """
    temperatures, weights = _validated_dem_weights(temperatures, weights)
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("coverage tolerance must be finite and in [0, 1)")
    active = weights > 0
    low = record["temperature_min_keV"]
    high = record["temperature_max_keV"]
    covered = (temperatures >= low) & (temperatures <= high)
    usable = active & covered
    if not np.any(active & ~covered):
        return usable
    label = f"{symbol} " if symbol else ""
    if not np.any(usable):
        outside = temperatures[active]
        raise ValueError(
            f"{label}temperature is outside complete triplet coverage: no overlap; "
            f"no positive-weight temperature falls in {low:.5f}-{high:.5f} keV "
            f"({outside.size} active bins span "
            f"{outside.min():.5f}-{outside.max():.5f} keV)")
    fraction = _coverage_bound(temperatures, weights, record, grid, active, covered)
    if not np.isfinite(fraction) or fraction > tolerance:
        outside = temperatures[active & ~covered]
        raise ValueError(
            f"{label}temperature is outside complete triplet coverage: "
            f"{outside.size} of {int(np.count_nonzero(active))} active DEM bins "
            f"span {outside.min():.5f}-{outside.max():.5f} keV, outside the "
            f"complete-triplet range {low:.5f}-{high:.5f} keV; the discarded "
            f"per-line integral bound {fraction:.6g} exceeds tolerance "
            f"{tolerance:.6g}; restrict the DEM to {low:.5f}-{high:.5f} keV")
    return usable
