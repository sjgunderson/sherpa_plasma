"""Print xsvwdem DEM metadata and triplet-coverage diagnostics.

Run with the same Python/Sherpa/XSPEC environment used by the failing fit:
    python diagnose_xsvwdem.py --root /path/to/apec/root
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sherpa.astro import xspec

from sherpa_plasma import wrap_triplet_pumping
from sherpa_plasma._uv_model import _capture_cie_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--redshift", type=float, default=0.02)
    parser.add_argument("--si", type=float, default=1.0)
    args = parser.parse_args()

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", str(Path(args.root).expanduser().resolve()))
    child = xspec.XSvwdem("diagnostic_xsvwdem")
    child.Tmax = args.tmax
    child.beta = args.beta
    child.Redshift = args.redshift
    child.Si = args.si
    wrapper = wrap_triplet_pumping(child, "Si")

    edges = np.geomspace(0.01, 100.0, 1001)
    from sherpa.astro.xspec import _xspec
    from sherpa_plasma import _xspec_bridge
    trial = np.array([parameter.val for parameter in child.pars], dtype=float)
    flux, temperatures, weights = _capture_cie_evaluation(
        child, type(child).__name__.lower(), trial, edges[:-1], edges[1:],
        _xspec_bridge, _xspec.__file__,
    )
    temperatures = np.asarray(temperatures)
    weights = np.asarray(weights)
    record = wrapper._entries[0][0]
    positive = weights > 0
    print(json.dumps({
        "xspec_version": xspec.get_xsversion(),
        "model_type": type(child).__name__,
        "parameters": {parameter.name: parameter.val for parameter in child.pars},
        "temperature_units": "keV (XSPEC model parameter convention)",
        "temperature_count": int(temperatures.size),
        "positive_count": int(np.count_nonzero(positive)),
        "temperature_min": float(temperatures.min()),
        "temperature_max": float(temperatures.max()),
        "positive_temperature_min": float(temperatures[positive].min()),
        "positive_temperature_max": float(temperatures[positive].max()),
        "positive_weight_sum": float(weights[positive].sum()),
        "si_coverage_keV": [record["temperature_min_keV"], record["temperature_max_keV"]],
        "active_outside_si": int(np.count_nonzero(
            positive & ((temperatures < record["temperature_min_keV"]) |
                       (temperatures > record["temperature_max_keV"]))
        )),
        "native_flux_finite": bool(np.isfinite(flux).all()),
    }, indent=2))
    wrapper.p_Si = 1.0
    try:
        wrapper(edges[:-1], edges[1:])
    except Exception as error:
        print(json.dumps({"wrapper_error": str(error)}))
    else:
        print(json.dumps({"wrapper_error": None}))


if __name__ == "__main__":
    main()
