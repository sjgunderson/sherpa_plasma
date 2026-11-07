"""Spectral coordinates, separate from counts and Sherpa's keV model grid.

Conversions operate on bin boundaries. Bin integrals are invariant; densities
must be formed using the *converted* widths, after any rebinning.
"""
import numpy as np

# Exact SI h, c, and elementary charge, expressed in these units.
HC_KEV_ANGSTROM = 12.398419843320026
H_KEV_S = 6.62607015e-34 / 1.602176634e-16

# Canonical name -> (coordinate family, scale to keV or Angstrom, label).
_SPECS = {
    "energy": ("energy", 1., "keV"),
    "eV": ("energy", 1.e-3, "eV"),
    "MeV": ("energy", 1.e3, "MeV"),
    "GeV": ("energy", 1.e6, "GeV"),
    "TeV": ("energy", 1.e9, "TeV"),
    "erg": ("energy", 1.e-7 / 1.602176634e-16, "erg"),
    "J": ("energy", 1. / 1.602176634e-16, "J"),
    "wavelength": ("wavelength", 1., "Angstrom"),
    "nm": ("wavelength", 10., "nm"),
    "um": ("wavelength", 1.e4, "um"),
    "mm": ("wavelength", 1.e7, "mm"),
    "cm": ("wavelength", 1.e8, "cm"),
    "m": ("wavelength", 1.e10, "m"),
    "Hz": ("frequency", H_KEV_S, "Hz"),
    "kHz": ("frequency", H_KEV_S * 1.e3, "kHz"),
    "MHz": ("frequency", H_KEV_S * 1.e6, "MHz"),
    "GHz": ("frequency", H_KEV_S * 1.e9, "GHz"),
    "THz": ("frequency", H_KEV_S * 1.e12, "THz"),
    "channel": ("channel", 1., "channel"),
}
_ALIASES = {name.lower(): name for name in _SPECS}
_ALIASES.update({
    "kev": "energy", "ener": "energy",
    "angstrom": "wavelength", "angstroms": "wavelength", "a": "wavelength",
    "å": "wavelength", "ångström": "wavelength", "angstroem": "wavelength",
    "wave": "wavelength", "micron": "um", "microns": "um",
    "µm": "um", "μm": "um", "frequency": "Hz", "freq": "Hz",
    "channels": "channel", "chan": "channel", "bin": "channel",
})


def supported_units():
    """Return supported display labels (energy/wavelength aliases also work)."""
    return tuple(spec[2] for spec in _SPECS.values())


def normalize_unit(value):
    try:
        return _ALIASES[value.strip().lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown spectral unit {value!r}; choose from "
                         + ", ".join(supported_units())) from None


def unit_family(value):
    return _SPECS[normalize_unit(value)][0]


def unit_label(value):
    return _SPECS[normalize_unit(value)][2]


def unit_mathtext(value):
    """A unit fragment for Matplotlib Mathtext, without dollar delimiters."""
    name = normalize_unit(value)
    if name == "wavelength":
        return r"\mathrm{\AA}"
    if name == "um":
        return r"\mu\mathrm{m}"
    return rf"\mathrm{{{unit_label(name)}}}"


def quantity_label(quantity, *powers):
    """Format a quantity with upright units and mathematical exponents.

    Each pair is a Mathtext unit fragment and its integer exponent.
    Zero powers are omitted, including cancelled density/power factors.
    """
    parts = [unit if power == 1 else rf"{{{unit}}}^{{{power}}}" if unit.startswith(r"\mu")
             else rf"{unit}^{{{power}}}"
             for unit, power in powers if power]
    return quantity if not parts else quantity + r" ($" + r"\,".join(parts) + r"$)"


def axis_label(value):
    family = unit_family(value)
    return "Channel" if family == "channel" else \
        family.capitalize() + r" ($" + unit_mathtext(value) + r"$)"


def convert_bins(lo, hi, source_unit, target_unit, *, hc=HC_KEV_ANGSTROM):
    """Convert increasing bins; return low/high edges and input permutation.

    No data interpolation occurs here. Apply the returned permutation to bin
    integrals/errors and to BOTH axes of a covariance matrix. Physical channel
    conversion needs a calibration and is deliberately not inferred.
    ``hc`` can match the constant used by the installed Sherpa release.
    """
    source, target = normalize_unit(source_unit), normalize_unit(target_unit)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    order = np.arange(lo.size)
    if source == target:
        return lo.copy(), hi.copy(), order
    sf, ss, _ = _SPECS[source]
    tf, ts, _ = _SPECS[target]
    if "channel" in (sf, tf):
        raise ValueError("channel coordinates cannot be converted without a response")
    reciprocal = (sf == "wavelength") != (tf == "wavelength")
    if not reciprocal:
        factor = ss / ts
        return lo * factor, hi * factor, order
    if np.any(lo <= 0):
        raise ValueError("energy/wavelength/frequency conversion requires positive bin boundaries")
    if not np.isfinite(hc) or hc <= 0:
        raise ValueError("hc must be finite and positive")
    order = order[::-1]
    factor = hc / (ss * ts)
    return factor / hi[order], factor / lo[order], order

