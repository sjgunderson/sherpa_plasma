"""Compare selected APEC line contributions against XSPEC in subprocesses.

Run with a CIAO Python and an explicit APEC root; see ``--help``. This tool
copies the line FITS and removes one transition at a time; it never changes
the source database. The difference is measured before response folding.
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


def _edges(args):
    if args.grid == "linear":
        return np.linspace(args.energy_min, args.energy_max, args.bins + 1)
    fraction = np.linspace(0., 1., args.bins + 1)
    fraction = fraction ** 2
    return args.energy_min + (args.energy_max - args.energy_min) * fraction


def _worker(args):
    from sherpa.astro import xspec
    from sherpa_plasma.plasma import _SYMBOLS

    xspec.set_xsxset("APECROOT", args.root)
    xspec.set_xsabund(args.abundance_table)
    if args.trace_abundance is not None:
        xspec.set_xsxset("APEC_TRACE_ABUND", args.trace_abundance)
    if args.log_interpolation:
        xspec.set_xsxset("APECLOGINTERP", "yes")
    model = getattr(xspec, "XS" + args.model)("reference")
    model.kT = args.temperature
    symbol = _SYMBOLS[args.element]
    if args.model == "apec":
        model.Abundanc = args.abundance
    elif hasattr(model, symbol):
        getattr(model, symbol).val = args.abundance
    else:
        model.Fe = args.abundance
    model.norm = args.norm
    model.Redshift = args.redshift
    if args.model == "bvvapec":
        model.Velocity = args.velocity
    edges = _edges(args)
    values = model(edges[:-1], edges[1:])
    if not np.isfinite(values).all() or not np.any(values > 0):
        raise RuntimeError("XSPEC returned an invalid or empty spectrum")
    print("UV_REFERENCE_RESULT " + json.dumps(values.tolist(), allow_nan=False))


def _spectrum(root, args):
    command = [
        sys.executable, __file__, "--worker", "--root", str(root),
        "--temperature", str(args.temperature), "--abundance", str(args.abundance),
        "--norm", str(args.norm), "--redshift", str(args.redshift),
        "--energy-min", str(args.energy_min), "--energy-max", str(args.energy_max),
        "--bins", str(args.bins),
        "--grid", args.grid,
        "--element", str(args.element),
        "--upper", *map(str, args.upper),
        "--abundance-table", args.abundance_table,
        "--model", args.model,
        "--velocity", str(args.velocity),
        "--profile", args.profile,
    ]
    if args.trace_abundance is not None:
        command.extend(("--trace-abundance", args.trace_abundance))
    if args.log_interpolation:
        command.append("--log-interpolation")
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"XSPEC worker failed:\n{process.stdout}\n{process.stderr}")
    marker = "UV_REFERENCE_RESULT "
    results = [line[len(marker):] for line in process.stdout.splitlines()
               if line.startswith(marker)]
    if len(results) != 1:
        raise RuntimeError(f"XSPEC worker did not report one result:\n{process.stdout}")
    return np.asarray(json.loads(results[0]), dtype=float)


def _run(args):
    from astropy.io import fits
    from sherpa_plasma._uv_lines import _interpolate_native_lines, _read_native_lines
    from sherpa_plasma._uv_projection import _unbroadened_line_bins
    from sherpa_plasma.plasma import LineID, _SYMBOLS
    from sherpa.astro import xspec

    root = Path(args.root).expanduser().resolve()
    line_file = Path(f"{root}_line.fits")
    continuum_file = Path(f"{root}_coco.fits")
    with fits.open(line_file, memmap=True) as hdus:
        native_abundance = hdus[0].header.get("SABUND_SOURCE", "").strip()
    if native_abundance != "AG89":
        raise ValueError(f"unverified APEC abundance source: {native_abundance!r}")
    symbol = _SYMBOLS[args.element]
    xspec.set_xsabund(args.abundance_table)
    reference_solar = xspec.get_xsabundances("angr")[symbol]
    selected_solar = xspec.get_xsabundances()[symbol]
    lines = tuple(LineID(args.element, args.element - 1, upper, 1)
                  for upper in args.upper)
    data = _read_native_lines(line_file, lines)
    linear, wavelengths = _interpolate_native_lines(data, args.temperature)
    baseline = _spectrum(root, args)
    with tempfile.TemporaryDirectory(prefix="sherpa-plasma-uv-") as directory:
        isolated_root = Path(directory) / "apec"
        shutil.copyfile(line_file, f"{isolated_root}_line.fits")
        os.symlink(continuum_file, f"{isolated_root}_coco.fits")
        selected = range(3) if args.transition is None else (args.transition,)
        for column in selected:
            line = lines[column]
            originals = []
            with fits.open(f"{isolated_root}_line.fits", mode="update", memmap=True) as hdus:
                for index, hdu in enumerate(hdus[2:], start=2):
                    rows = np.flatnonzero(line.mask(hdu.data))
                    if len(rows):
                        originals.append((index, rows, hdu.data["Epsilon"][rows].copy()))
                        hdu.data["Epsilon"][rows] = 0.0
                hdus.flush()
            without = _spectrum(isolated_root, args)
            difference = baseline - without
            main_elements = {"C", "N", "O", "Ne", "Mg", "Al", "Si", "S", "Ar",
                             "Ca", "Fe", "Ni"}
            is_trace = args.model in ("apec", "vapec") and symbol not in main_elements
            if not is_trace or args.trace_abundance == "Fe":
                element_abundance = args.abundance
            elif args.trace_abundance is None:
                element_abundance = 1.
            else:
                element_abundance = float(args.trace_abundance)
            expected = (linear[column] * 1.e14 * element_abundance * args.norm
                        * selected_solar / reference_solar / (1 + args.redshift))
            edges = _edges(args)
            energy = 12.39841974 / wavelengths[column] / (1 + args.redshift)
            predicted_bin = int(_unbroadened_line_bins(
                edges[:-1], edges[1:], [wavelengths[column]], args.redshift
            )[0])
            observed_bin = int(np.argmax(difference))
            profile_error = None
            if expected == 0:
                if np.any(np.abs(difference) > 1.e-18):
                    raise AssertionError(f"{line}: XSPEC line is nonzero at zero abundance")
                observed_bin = None
            elif args.model == "bvvapec":
                if args.profile != "measure":
                    sigma = energy * args.velocity / 299792.458
                    if args.profile == "xspec":
                        gaussian = xspec.XSgaussian("profile_reference")
                        gaussian.LineE = energy
                        gaussian.Sigma = sigma
                        gaussian.norm = 1.
                        profile = gaussian(edges[:-1], edges[1:])
                    else:
                        from scipy.special import ndtr

                        profile = np.diff(ndtr((edges - energy) / sigma))
                    error = float(np.max(np.abs(difference - expected * profile)) / expected)
                    profile_error = error
                    if error > args.rtol:
                        raise AssertionError(f"{line}: maximum profile error / total flux "
                                             f"{error} exceeds {args.rtol}")
            else:
                if not np.isclose(difference.sum(), expected, rtol=args.rtol, atol=1.e-18):
                    raise AssertionError(f"{line}: XSPEC line flux {difference.sum()} "
                                         f"does not match expected {expected}")
                if np.count_nonzero(difference) != 1 or difference.min() < 0:
                    raise AssertionError(f"{line}: line deposition is not a single nonnegative bin")
                if predicted_bin != observed_bin:
                    raise AssertionError(
                        f"{line}: XSPEC bin {observed_bin} differs from predicted "
                        f"bin {predicted_bin} at {energy} keV"
                    )
            print(json.dumps({
                "transition": str(line),
                "isolated_flux": float(difference.sum()),
                "nonzero_bins": int(np.count_nonzero(difference)),
                "linear_native_emissivity": float(linear[column]),
                "flux_per_emissivity": float(difference.sum() / linear[column]),
                "minimum_difference": float(difference.min()),
                "observed_bin": observed_bin,
                "predicted_energy_keV": float(energy),
                "wavelength_angstrom": float(wavelengths[column]),
                "profile_max_error_relative_total": profile_error,
            }))
            with fits.open(f"{isolated_root}_line.fits", mode="update", memmap=True) as hdus:
                for index, rows, values in originals:
                    hdus[index].data["Epsilon"][rows] = values
                hdus.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="APEC FITS root without _line.fits")
    parser.add_argument("--temperature", type=float, default=0.43)
    parser.add_argument("--element", type=int, default=11, choices=range(3, 31))
    parser.add_argument("--upper", type=int, nargs=3, default=(2, 4, 5),
                        help="forbidden, J=1 and J=2 upper level indices")
    parser.add_argument("--transition", type=int, choices=(0, 1, 2),
                        help="isolate only one transition (0=forbidden, 1 or 2=intercombination)")
    parser.add_argument("--abundance", type=float, default=1.0)
    parser.add_argument("--abundance-table", default="angr")
    parser.add_argument("--model", choices=("apec", "vapec", "vvapec", "bvvapec"),
                        default="vvapec")
    parser.add_argument("--velocity", type=float, default=0.,
                        help="bvvapec Gaussian velocity sigma in km/s")
    parser.add_argument("--profile", choices=("analytic", "xspec", "measure"), default="analytic",
                        help="reference profile, or measure without assuming a profile")
    parser.add_argument("--trace-abundance",
                        help="APEC_TRACE_ABUND value: a number or Fe")
    parser.add_argument("--norm", type=float, default=1.0)
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--energy-min", type=float, default=1.0)
    parser.add_argument("--energy-max", type=float, default=1.2)
    parser.add_argument("--bins", type=int, default=2000)
    parser.add_argument("--grid", choices=("linear", "quadratic"), default="linear")
    parser.add_argument("--rtol", type=float, default=1.e-9)
    parser.add_argument("--log-interpolation", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    numbers = (args.temperature, args.abundance, args.norm, args.redshift,
               args.energy_min, args.energy_max, args.rtol, args.velocity)
    if not all(np.isfinite(value) for value in numbers):
        parser.error("numeric options must be finite")
    if args.bins < 2 or args.energy_min <= 0 or args.energy_max <= args.energy_min:
        parser.error("require at least two bins and 0 < energy-min < energy-max")
    if args.temperature <= 0 or args.rtol <= 0:
        parser.error("temperature and tolerance must be positive")
    if args.velocity < 0 or (args.velocity and args.model != "bvvapec"):
        parser.error("velocity must be nonnegative and requires bvvapec")
    if args.model == "bvvapec" and args.velocity == 0 and args.profile != "measure":
        parser.error("zero-velocity bvvapec requires --profile measure; "
                     "its intrinsic profile is not yet modeled")
    if args.abundance <= 0 or args.norm <= 0 or args.redshift <= -1:
        parser.error("require positive abundance/norm and redshift > -1")
    if args.trace_abundance is not None and args.trace_abundance != "Fe":
        try:
            trace = float(args.trace_abundance)
        except ValueError:
            parser.error("trace-abundance must be a nonnegative number or Fe")
        if not np.isfinite(trace) or trace < 0:
            parser.error("trace-abundance must be a nonnegative number or Fe")
    if args.worker:
        _worker(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()

