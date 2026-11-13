"""Native XSET regressions against independently isolated XSPEC lines.

Requires Sherpa/XSPEC, both rebuilt bridges, and an explicit reference APEC root.
No scientific backend is replaced with a test double in this file.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_uv_model_real import _require_dem_reference

ROOT = os.environ.get("SHERPA_PLASMA_UV_XSPEC_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="requires an explicit APEC root")


@pytest.mark.parametrize("kind,thermal,threaded", [
    ("apec", "yes", "no"),
    ("bapec", "no", "yes"),
    ("bvapec", "yes", "yes"),
    ("bvvapec", "yes", "no"),
    ("wdem", "yes", "yes"),
    ("bvwdem", "no", "yes"),
    ("bvvwdem", "yes", "yes"),
    ("mkcflow", "yes", "yes"),
])
def test_xset_pumping_matches_isolated_native_profiles(kind, thermal, threaded):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    settings = {"APECTHERMAL": thermal, "APECMULTITHREAD": threaded}
    parameters = {"Redshift": .02, "norm": .3}
    if "wdem" in kind:
        parameters.update(Tmax=2.1, beta=.5, nH=1., switch=2)
    elif kind == "mkcflow":
        parameters.update(lowT=.5, highT=2.1, switch=2)
    else:
        parameters["kT"] = 1.
    if kind.startswith("b"):
        parameters["Velocity"] = 350.
    parameters["Si" if kind in ("bvapec", "bvvapec", "bvwdem", "bvvwdem")
               else "Abundanc"] = .6
    command = [
        sys.executable, str(Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"),
        "--root", ROOT, "--model", kind, "--parameters", json.dumps(parameters),
        "--xset", json.dumps(settings), "--element", "14", "--include-profile",
        "--energy-min", "1.75", "--energy-max", "1.95", "--bins", "8000",
    ]
    if kind == "mkcflow":
        command.append("--fixed-cooling-weights")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(row) for row in result.stdout.splitlines() if row.startswith("{")]
    assert len(lines) == 3 and all(row["weights_unchanged"] for row in lines)
    saved = xspec.get_xsxset()
    abundance = xspec.get_xsabundances()
    table = xspec.get_xsabund()
    try:
        xspec.set_xsabund("angr")
        xspec.set_xsxset("APECROOT", ROOT)
        for name, value in settings.items():
            xspec.set_xsxset(name, value)
        child = getattr(xspec, "XS" + kind)("native_xset_" + kind)
        for name, value in parameters.items():
            getattr(child, name).val = value
        wrapper = wrap_triplet_pumping(child, "Si")
        edges = np.linspace(1.75, 1.95, 8001)
        bare = child(edges[:-1], edges[1:])
        np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]), bare)
        wrapper.p_Si = 1.
        actual = wrapper(edges[:-1], edges[1:]) - bare
        f, i1, i2 = (row["predicted_flux"] for row in lines)
        transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
        eta1, eta2 = wrapper._entries[0][0]["calibration"]["redistribution"]
        profiles = [np.asarray(row["isolated_profile"]) for row in lines]
        expected = transfer * (-profiles[0] / f + eta1 * profiles[1] / i1
                               + eta2 * profiles[2] / i2)
        np.testing.assert_allclose(actual, expected, rtol=3.e-7, atol=1.e-17)
        assert actual.sum() == pytest.approx(0., abs=1.e-16)
    finally:
        for name in ("APECROOT", *settings):
            xspec.set_xsxset(name, saved.get(name, ""))
        if table == "file":
            xspec.set_xsabundances(abundance)
        else:
            xspec.set_xsabund(table)


def test_real_multithreaded_wrapper_matches_serial_and_native_controls():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import _xspec_profiles, wrap_triplet_pumping
    from sherpa_plasma._uv_model import _load_broad_profiles, _temperature_interval

    saved = xspec.get_xsxset()
    spectra = []
    try:
        xspec.set_xsxset("APECROOT", ROOT)
        xspec.set_xsxset("APECTHERMAL", "yes")
        child = xspec.XSbwdem("threaded_comparison")
        child.Tmax, child.beta, child.Redshift, child.Velocity = 2.1, .5, .02, 350.
        edges = np.linspace(1.75, 1.95, 8001)
        for threaded in ("no", "yes"):
            xspec.set_xsxset("APECMULTITHREAD", threaded)
            wrapper = wrap_triplet_pumping(child, "Si")
            wrapper.p_Si = 1.
            record, grid, _ = wrapper._entries[0]
            calibration = record["calibration"]
            profiles = _load_broad_profiles(
                wrapper._library_path, ROOT, 14, calibration["ground"],
                (calibration["metastable"], *calibration["triplet_p"][1:]),
                *_temperature_interval(grid), wrapper._manifest["line_file_sha256"],
                wrapper._manifest["continuum_file_sha256"], wrapper._data_identities[:2],
                tuple(grid.wavelengths[0]))
            assert _xspec_profiles._control_options(profiles, 350., True) == (
                True, threaded == "yes", 350.)
            before = _xspec_profiles._snapshot_db()
            _xspec_profiles.project(profiles, edges, [1.], [1.], .02, 350., True)
            assert _xspec_profiles._snapshot_db() == before
            spectra.append(wrapper(edges[:-1], edges[1:]))
        np.testing.assert_allclose(spectra[1], spectra[0], rtol=2.e-12, atol=1.e-20)
    finally:
        for name in ("APECROOT", "APECTHERMAL", "APECMULTITHREAD"):
            xspec.set_xsxset(name, saved.get(name, ""))
