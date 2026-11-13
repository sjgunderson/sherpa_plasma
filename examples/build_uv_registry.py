"""Build a compact calibration manifest from a previously generated inventory.

The manifest contains candidates, not an assertion that XSPEC compatibility
has been validated. Always pair it with the independent reference tests.
"""

import argparse
import hashlib
import json
from pathlib import Path

from sherpa_plasma._uv_lines import _read_native_lines
from sherpa_plasma.plasma import LineID


def build_registry(inventory, line_file, continuum_file, version):
    digest = hashlib.sha256()
    with open(line_file, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != inventory["line_file_sha256"]:
        raise ValueError("line file does not match the inventory SHA256")
    continuum_digest = hashlib.sha256()
    with open(continuum_file, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            continuum_digest.update(chunk)
    result = {
        "version": version,
        "line_file_sha256": inventory["line_file_sha256"],
        "continuum_file_sha256": continuum_digest.hexdigest(),
        "filemap_sha256": inventory["filemap_sha256"],
        "ions": {},
    }
    for z, candidate in inventory["ions"].items():
        if candidate["status"] != "candidate_unvalidated":
            continue
        z = int(z)
        c = candidate["calibration"]
        upper = (c["metastable"], *c["triplet_p"][1:])
        lines = [LineID(z, z - 1, value, c["ground"]) for value in upper]
        grid = _read_native_lines(line_file, lines)
        indices = candidate["complete_indices"]
        reference = (5.e6 * 8.617333262145e-8 if z == 11
                     else float(grid.temperatures[indices[len(indices) // 2]]))
        result["ions"][str(z)] = {
            "calibration": c,
            "reference_temperature_keV": reference,
            "level_file": candidate["level_file"],
            "level_file_sha256": candidate["level_file_sha256"],
            "radiative_file": candidate["radiative_file"],
            "radiative_file_sha256": candidate["radiative_file_sha256"],
            "temperature_min_keV": candidate["temperature_min_keV"],
            "temperature_max_keV": candidate["temperature_max_keV"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--line-file", type=Path, required=True)
    parser.add_argument("--continuum-file", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    with args.inventory.open(encoding="utf-8") as file:
        inventory = json.load(file)
    print(json.dumps(build_registry(inventory, args.line_file, args.continuum_file, args.version),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
