"""Compare isolated XSPEC DEM triplet lines with captured-temperature predictions.

The reference APEC line file is copied before individual lines are removed.
Each native spectrum is evaluated in a fresh process to avoid XSPEC data caches.
This is a diagnostic, not validation of a UV-pumping model.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _worker(args):
    from sherpa.astro import xspec
    from sherpa.astro.xspec import _xspec
    from sherpa_plasma import _xspec_bridge
    from sherpa_plasma.xspec_capture import _capture_cie_evaluation

    xspec.set_xsxset("APECROOT", args.root)
    xspec.set_xsabund(args.abundance_table)
    if args.fixed_weights is not None:
        settings = json.loads(args.fixed_weights)
        apec = xspec.XSvvapec("fixed_dem_reference")
        apec.Redshift = settings["redshift"]
        apec.norm = settings["norm"]
        for symbol in ("C", "N", "O", "Ne", "Na", "Mg", "Al", "Si", "S",
                       "Ar", "Ca", "Fe", "Ni"):
            getattr(apec, symbol).val = settings["abundance"]
        edges = np.linspace(args.energy_min, args.energy_max, args.bins + 1)
        flux = np.zeros(args.bins)
        for temperature, weight in zip(settings["temperatures"], settings["weights"]):
            apec.kT = temperature
            flux += weight * apec(edges[:-1], edges[1:])
        print("UV_DEM_RESULT " + json.dumps({"flux": flux.tolist()}, allow_nan=False))
        return
    model = getattr(xspec, "XS" + args.model)("dem_characterization")
    available = {par.name.lower(): par for par in model.pars}
    for name, value in json.loads(args.parameters).items():
        if name.lower() not in available:
            raise ValueError(f"{args.model} has no parameter {name!r}")
        available[name.lower()].val = float(value)
    edges = np.linspace(args.energy_min, args.energy_max, args.bins + 1)
    trial = np.array([par.val for par in model.pars], dtype=float)
    flux, temperatures, weights = _capture_cie_evaluation(
        model, type(model).__name__.lower(), trial, edges[:-1], edges[1:],
        _xspec_bridge, _xspec.__file__,
    )
    print("UV_DEM_RESULT " + json.dumps({
        "parameters": {par.name: float(value) for par, value in zip(model.pars, trial)},
        "flux": flux.tolist(),
        "temperatures": list(temperatures),
        "weights": list(weights),
    }, allow_nan=False))


def _evaluate(root, args, fixed_weights=None):
    command = [
        sys.executable, __file__, "--worker", "--root", str(root),
        "--model", args.model, "--parameters", args.parameters,
        "--abundance-table", args.abundance_table,
        "--energy-min", str(args.energy_min), "--energy-max", str(args.energy_max),
        "--bins", str(args.bins),
    ]
    if fixed_weights is not None:
        command.extend(("--fixed-weights", json.dumps(fixed_weights)))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"XSPEC worker failed:\n{result.stdout}\n{result.stderr}")
    marker = "UV_DEM_RESULT "
    rows = [line[len(marker):] for line in result.stdout.splitlines()
            if line.startswith(marker)]
    if len(rows) != 1:
        raise RuntimeError(f"XSPEC worker did not report one result:\n{result.stdout}")
    return json.loads(rows[0])


def _run(args):
    from astropy.io import fits
    from sherpa.astro import xspec
    from sherpa_plasma._uv_lines import _interpolate_native_lines, _read_native_lines
    from sherpa_plasma._uv_model import _file_identity, _file_sha256, _MANIFESTS
    from sherpa_plasma.plasma import LineID, _SYMBOLS
    import sherpa_plasma

    root = Path(args.root).expanduser().resolve()
    line_file = Path(str(root) + "_line.fits")
    continuum_file = Path(str(root) + "_coco.fits")
    hashes = tuple(_file_sha256(path, _file_identity(path))
                   for path in (line_file, continuum_file))
    for filename in _MANIFESTS:
        manifest = json.loads((Path(sherpa_plasma.__file__).parent / filename).read_text())
        if hashes == (manifest["line_file_sha256"], manifest["continuum_file_sha256"]):
            break
    else:
        raise ValueError("APEC files do not match a verified UV calibration")
    record = manifest["ions"].get(str(args.element))
    if record is None:
        raise ValueError("element has no verified complete-triplet calibration")
    calibration = record["calibration"]
    lines = tuple(LineID(args.element, args.element - 1, upper, calibration["ground"])
                  for upper in (calibration["metastable"], *calibration["triplet_p"][1:]))
    grid = _read_native_lines(line_file, lines)
    baseline = _evaluate(root, args)
    temperatures = np.asarray(baseline["temperatures"], dtype=float)
    weights = np.asarray(baseline["weights"], dtype=float)
    if (temperatures.ndim != 1 or temperatures.shape != weights.shape
            or not np.isfinite(temperatures).all() or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("captured temperature weights are invalid")
    active = weights > 0
    emissivities = np.array([
        _interpolate_native_lines(grid, temperature)[0]
        for temperature in temperatures[active]
    ])
    if not len(emissivities):
        raise ValueError("captured DEM has no nonzero weights")
    weighted_emissivities = weights[active] @ emissivities
    params = {name.lower(): value for name, value in baseline["parameters"].items()}
    if params["switch"] != 2:
        raise ValueError("only switch=2 (AtomDB) can be compared with APEC line files")
    xspec.set_xsabund(args.abundance_table)
    symbol = _SYMBOLS[args.element]
    solar_scale = xspec.get_xsabundances()[symbol] / xspec.get_xsabundances("angr")[symbol]
    scalar_elements = {"C", "N", "O", "Ne", "Na", "Mg", "Al", "Si", "S",
                       "Ar", "Ca", "Fe", "Ni"}
    if args.model in ("vwdem", "vvwdem", "bvwdem", "bvvwdem"):
        abundance = params.get(symbol.lower(), 1.)
    else:
        abundance = params["abundanc"] if symbol in scalar_elements else 1.
    predicted = (weighted_emissivities * 1.e14 * solar_scale * abundance
                 * params["norm"] / (1 + params["redshift"]))
    base_flux = np.asarray(baseline["flux"])
    fixed_weights = None
    if args.fixed_cooling_weights:
        if args.model not in ("coolflow", "mkcflow"):
            raise ValueError("fixed cooling-weight reference requires a cooling-flow model")
        fixed_weights = {
            "temperatures": temperatures.tolist(), "weights": weights.tolist(),
            "redshift": params["redshift"], "norm": params["norm"],
            "abundance": params["abundanc"],
        }
        reference = np.asarray(_evaluate(root, args, fixed_weights)["flux"])
        if not np.allclose(reference, base_flux, rtol=2.e-11, atol=1.e-18):
            raise ValueError("fixed-weight APEC sum does not reproduce the cooling-flow spectrum")
        base_flux = reference
    with tempfile.TemporaryDirectory(prefix="sherpa-plasma-dem-") as directory:
        temporary_root = Path(directory) / "apec"
        shutil.copyfile(line_file, str(temporary_root) + "_line.fits")
        os.symlink(continuum_file, str(temporary_root) + "_coco.fits")
        for index, line in enumerate(lines):
            originals = []
            with fits.open(str(temporary_root) + "_line.fits", mode="update") as hdus:
                for hdu_index, hdu in enumerate(hdus[2:], start=2):
                    rows = np.flatnonzero(line.mask(hdu.data))
                    if len(rows):
                        originals.append((hdu_index, rows, hdu.data["Epsilon"][rows].copy()))
                        hdu.data["Epsilon"][rows] = 0.
                hdus.flush()
            modified = _evaluate(temporary_root, args, fixed_weights)
            difference = base_flux - np.asarray(modified["flux"])
            same_weights = (True if fixed_weights is not None else
                            (np.array_equal(temperatures, modified["temperatures"])
                             and np.array_equal(weights, modified["weights"])))
            result = {
                "transition": str(line),
                "model": args.model,
                "parameters": baseline["parameters"],
                "temperature_count": len(temperatures),
                "temperature_min_keV": float(temperatures[active].min()),
                "temperature_max_keV": float(temperatures[active].max()),
                "weight_sum": float(weights.sum()),
                "weights_unchanged": bool(same_weights),
                "fixed_weight_reference": fixed_weights is not None,
                "isolated_flux": float(difference.sum()),
                "predicted_flux": float(predicted[index]),
                "relative_difference": (float((difference.sum() - predicted[index])
                                              / predicted[index])
                                        if predicted[index] != 0 else None),
                "nonzero_bins": int(np.count_nonzero(difference)),
                "observed_bin": int(np.argmax(difference)),
            }
            if args.include_profile:
                result["isolated_profile"] = difference.tolist()
            print(json.dumps(result, allow_nan=False))
            with fits.open(str(temporary_root) + "_line.fits", mode="update") as hdus:
                for hdu_index, rows, values in originals:
                    hdus[hdu_index].data["Epsilon"][rows] = values
                hdus.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="verified APEC FITS root")
    parser.add_argument("--model", choices=("wdem", "vwdem", "vvwdem", "bwdem",
                                            "bvwdem", "bvvwdem", "mkcflow",
                                            "coolflow"), default="wdem")
    parser.add_argument("--parameters", default='{"Tmax": 0.8, "beta": 0.5}',
                        help="JSON object of Sherpa model parameter values")
    parser.add_argument("--element", type=int, default=11)
    parser.add_argument("--abundance-table", default="angr")
    parser.add_argument("--energy-min", type=float, default=1.)
    parser.add_argument("--energy-max", type=float, default=1.2)
    parser.add_argument("--bins", type=int, default=2000)
    parser.add_argument("--include-profile", action="store_true",
                        help="include the full isolated line profile in each result")
    parser.add_argument("--fixed-cooling-weights", action="store_true",
                        help="compare isolated APEC sums using original cooling-flow weights")
    parser.add_argument("--fixed-weights", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        parameters = json.loads(args.parameters)
    except json.JSONDecodeError as exc:
        parser.error(f"parameters must be a JSON object: {exc}")
    if not isinstance(parameters, dict):
        parser.error("parameters must be a JSON object")
    if (args.bins < 2 or not np.isfinite([args.energy_min, args.energy_max]).all()
            or args.energy_min <= 0 or args.energy_min >= args.energy_max):
        parser.error("require at least two bins and 0 < energy-min < energy-max")
    if args.worker:
        _worker(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()

