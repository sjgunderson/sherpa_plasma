"""Independent XSPEC reference checks for selected Na X lines."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = os.environ.get("SHERPA_PLASMA_UV_XSPEC_ROOT")
SCRIPT = Path(__file__).parents[1] / "examples" / "validate_uv_xspec.py"


@pytest.mark.skipif(not ROOT, reason="requires an explicit XSPEC APEC file root")
@pytest.mark.parametrize("extra", [
    [],
    ["--redshift", "0.1", "--abundance", "0.4", "--norm", "0.02",
     "--energy-min", "0.9", "--energy-max", "1.1"],
    ["--abundance-table", "aspl"],
    ["--abundance-table", "wilm"],
    ["--grid", "quadratic"],
    ["--element", "6", "--upper", "2", "5", "6", "--temperature", "0.1",
     "--energy-min", "0.29", "--energy-max", "0.32"],
    ["--element", "7", "--upper", "2", "5", "6", "--temperature", "0.15",
     "--energy-min", "0.41", "--energy-max", "0.44"],
    ["--element", "8", "--upper", "2", "5", "6", "--temperature", "0.2",
     "--energy-min", "0.55", "--energy-max", "0.58"],
    ["--element", "10", "--upper", "2", "5", "6", "--temperature", "0.4",
     "--energy-min", "0.89", "--energy-max", "0.93"],
    ["--element", "12", "--upper", "2", "5", "6", "--temperature", "0.8",
     "--energy-min", "1.31", "--energy-max", "1.36"],
    ["--model", "apec", "--abundance", "0.4"],
    ["--model", "apec", "--abundance", "0.4", "--trace-abundance", "0.7"],
    ["--model", "apec", "--abundance", "0.4", "--trace-abundance", "Fe"],
    ["--model", "vapec", "--abundance", "0.4"],
    ["--model", "vapec", "--abundance", "0.4", "--trace-abundance", "0.7"],
    ["--model", "vapec", "--abundance", "0.4", "--trace-abundance", "Fe"],
    ["--model", "vapec", "--abundance", "0.4", "--trace-abundance", "0"],
    ["--model", "apec", "--abundance", "0.4", "--element", "8",
     "--upper", "2", "5", "6", "--temperature", "0.2",
     "--energy-min", "0.55", "--energy-max", "0.58"],
    ["--model", "vapec", "--abundance", "0.4", "--element", "8",
     "--upper", "2", "5", "6", "--temperature", "0.2",
     "--energy-min", "0.55", "--energy-max", "0.58"],
])
def test_selected_line_flux_against_xspec(extra):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", ROOT, *extra],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('"isolated_flux"') == 3
