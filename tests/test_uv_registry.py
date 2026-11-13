"""Tests for the compact triplet registry builder."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_registry_does_not_require_critical_rates(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "examples" / "build_uv_registry.py"
    spec = importlib.util.spec_from_file_location("build_uv_registry", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    line = tmp_path / "line.fits"
    continuum = tmp_path / "continuum.fits"
    line.write_bytes(b"line")
    continuum.write_bytes(b"continuum")
    monkeypatch.setattr(builder, "_read_native_lines", lambda *args: SimpleNamespace(
        temperatures=np.array([.1, .2, .3])))
    candidate = {
        "status": "candidate_unvalidated",
        "calibration": {"metastable": 2, "triplet_p": [3, 4, 5], "ground": 1},
        "complete_indices": [0, 1, 2],
        "level_file": "levels", "level_file_sha256": "levelhash",
        "radiative_file": "rates", "radiative_file_sha256": "ratehash",
        "temperature_min_keV": .1, "temperature_max_keV": .3,
    }
    inventory = {
        "line_file_sha256": hashlib.sha256(b"line").hexdigest(),
        "filemap_sha256": "filemaphash",
        "ions": {"8": candidate},
    }
    registry = builder.build_registry(inventory, line, continuum, "test")
    record = registry["ions"]["8"]
    assert record["reference_temperature_keV"] == .2
    assert "reference_phi_c_s-1" not in record
    assert "reference_R0" not in record
    assert registry["continuum_file_sha256"] == hashlib.sha256(b"continuum").hexdigest()
