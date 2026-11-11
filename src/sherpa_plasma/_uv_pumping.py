"""Numerical primitives for AtomDB-backed He-like triplet pumping."""

import math


def _transfer_triplet_photons(forbidden, intercombination1, intercombination2, p):
    """Return photons transferred from forbidden to intercombination lines.

    All fluxes use the same intrinsic photon units; p is dimensionless.
    """
    values = tuple(float(value) for value in (
        forbidden, intercombination1, intercombination2, p,
    ))
    forbidden, intercombination1, intercombination2, p = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fluxes and p must be finite")
    if min(values) < 0:
        raise ValueError("fluxes and p must be nonnegative")

    scale = max(forbidden, intercombination1, intercombination2)
    if scale == 0 or forbidden == 0 or p == 0:
        return 0.0
    i_scale = max(intercombination1, intercombination2)
    if i_scale == 0:
        return 0.0

    # Delta/F = x/(1+x), where x = p * I/(F+I).
    # Keep the exponent of I/(F+I) separate even when that ratio underflows.
    total_scaled = (forbidden / scale + intercombination1 / scale
                    + intercombination2 / scale)
    i_scaled = intercombination1 / i_scale + intercombination2 / i_scale
    pump_m, pump_e = math.frexp(p)
    i_m, i_e = math.frexp(i_scale)
    scale_m, scale_e = math.frexp(scale)
    mantissa = (pump_m * (i_m / scale_m)
                * (i_scaled / total_scaled))
    exponent = pump_e + i_e - scale_e
    if exponent > 60:
        return forbidden
    if exponent < -1020:
        forbidden_m, forbidden_e = math.frexp(forbidden)
        return math.ldexp(forbidden_m * mantissa, forbidden_e + exponent)
    x = math.ldexp(mantissa, exponent)
    return forbidden * (x / (1.0 + x))
