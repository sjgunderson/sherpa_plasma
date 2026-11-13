"""Inventory He-like atomic inputs and explicit APEC triplet coverage.

The supplied filemap must match the APEC release. Atomic files are read from
the supplied data root; no files are downloaded by this script.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from sherpa_plasma._uv_atomic import _read_radiative_calibration
from sherpa_plasma._uv_lines import _read_native_lines
from sherpa_plasma.plasma import LineID


def _filemap(path):
    entries = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                kind, element, ion = map(int, parts[:3])
            except ValueError:
                continue
            if kind not in (2, 3) or not 3 <= element <= 30 or ion != element - 1:
                continue
            prefix = "$ATOMDB/"
            if not parts[3].startswith(prefix):
                raise ValueError(f"unsupported filemap path: {parts[3]}")
            relative = Path(parts[3][len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe filemap path: {relative}")
            mapped = entries.setdefault(element, {})
            if kind in mapped:
                raise ValueError(f"duplicate filemap entry for element {element}, type {kind}")
            mapped[kind] = relative
    return entries


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_path(root, relative):
    path = root / relative
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(f"atomic input not found: {path} or {compressed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filemap", required=True, type=Path)
    parser.add_argument("--atomic-root", required=True, type=Path)
    parser.add_argument("--line-file", required=True, type=Path)
    args = parser.parse_args()
    entries = _filemap(args.filemap)
    result = {
        "line_file_sha256": _hash(args.line_file),
        "filemap_sha256": _hash(args.filemap),
        "ions": {},
    }
    for z in range(3, 31):
        mapped = entries.get(z, {})
        if set(mapped) != {2, 3}:
            result["ions"][str(z)] = {"status": "missing_filemap_entry"}
            continue
        try:
            level = _atomic_path(args.atomic_root, mapped[2])
            radiative = _atomic_path(args.atomic_root, mapped[3])
            calibration = _read_radiative_calibration(level, radiative)
            lines = tuple(LineID(z, z - 1, upper, calibration.ground)
                          for upper in (calibration.metastable, *calibration.triplet_p[1:]))
            grid = _read_native_lines(args.line_file, lines)
            complete = np.isfinite(grid.emissivities).all(axis=1)
            indices = np.flatnonzero(complete)
            if not len(indices):
                raise ValueError("no complete native triplet")
            result["ions"][str(z)] = {
                "status": "candidate_unvalidated",
                "calibration": asdict(calibration),
                "level_file": str(mapped[2]),
                "radiative_file": str(mapped[3]),
                "level_file_sha256": _hash(level),
                "radiative_file_sha256": _hash(radiative),
                "complete_indices": indices.tolist(),
                "temperature_min_keV": float(grid.temperatures[indices[0]]),
                "temperature_max_keV": float(grid.temperatures[indices[-1]]),
                "complete_count": len(indices),
            }
        except (ValueError, FileNotFoundError) as exc:
            result["ions"][str(z)] = {"status": "unavailable", "reason": str(exc)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
