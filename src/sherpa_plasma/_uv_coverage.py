"""Validation of the complete captured DEM, independent of Sherpa and XSPEC."""

import numpy as np


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
