"""Sparse projection for the verified unbroadened XSPEC line profile."""

import numpy as np

# Numerics::KEVTOA in the CIAO 4.18 XSPEC headers, not the package's
# current-CODATA display-unit conversion constant.
_XSPEC_KEV_ANGSTROM = 12.39841974


def _unbroadened_line_bins(lo, hi, wavelengths, redshift):
    """Return one bin index per line, or -1 when the line is off-grid.

    Near an exact bin edge, floating-point operation ordering in XSPEC has
    not yet been validated, so those cases are rejected rather than silently
    subtracting photons from a potentially different bin.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    redshift = float(redshift)
    if (lo.ndim != 1 or hi.ndim != 1 or len(lo) < 2 or len(lo) != len(hi)
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo <= 0) or np.any(hi <= lo)
            or not np.array_equal(hi[:-1], lo[1:])):
        raise ValueError("XSPEC projection requires at least two contiguous ordered energy bins")
    if (wavelengths.ndim != 1 or not np.isfinite(wavelengths).all()
            or np.any(wavelengths <= 0)
            or not np.isfinite(redshift) or redshift <= -1):
        raise ValueError("wavelengths must be positive and redshift must be finite and greater than -1")
    edges = np.concatenate((lo[:1], hi))
    energies = _XSPEC_KEV_ANGSTROM / wavelengths / (1 + redshift)
    positions = np.searchsorted(edges, energies, side="left")
    nearest = np.clip(positions, 0, len(edges) - 1)
    previous = np.maximum(nearest - 1, 0)
    distance = np.minimum(np.abs(edges[nearest] - energies),
                          np.abs(edges[previous] - energies))
    tolerance = 16 * np.spacing(np.maximum(np.abs(energies), 1.0))
    if np.any(distance <= tolerance):
        raise ValueError("line energy is too close to a bin edge for validated XSPEC projection")
    bins = np.searchsorted(edges, energies, side="right") - 1
    bins[(bins < 0) | (bins >= len(lo))] = -1
    return bins
