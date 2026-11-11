"""Compare the experimental composite with independently isolated XSPEC lines."""

import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = os.environ.get("SHERPA_PLASMA_UV_XSPEC_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="requires an explicit APEC root")


@pytest.fixture(scope="module")
def isolated_sodium():
    script = Path(__file__).parents[1] / "examples" / "validate_uv_xspec.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]


def _model():
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSvvapec("test_plasma")
    child.kT = .43
    return child, wrap_triplet_pumping(child, ["Na"])


def _require_dem_reference():
    from sherpa_plasma._uv_model import _file_identity, _file_sha256, _MANIFESTS
    import sherpa_plasma

    paths = [Path(ROOT + suffix) for suffix in ("_line.fits", "_coco.fits")]
    hashes = [_file_sha256(path, _file_identity(path)) for path in paths]
    for filename in _MANIFESTS:
        manifest_path = Path(sherpa_plasma.__file__).parent / filename
        manifest = json.loads(manifest_path.read_text())
        if hashes == [manifest["line_file_sha256"], manifest["continuum_file_sha256"]]:
            return
    pytest.skip("DEM reference tests require a bundled calibration")


@pytest.mark.parametrize("model,parameters", [
    ("wdem", {"Tmax": .8, "beta": .5, "Redshift": .02, "abundanc": .6}),
    ("wdem", {"Tmax": .7, "beta": .7, "Redshift": .05, "abundanc": 1.4,
              "inv_slope": .5, "norm": .3}),
    ("mkcflow", {"lowT": .25, "highT": .8, "Redshift": .02, "Abundanc": .6}),
])
def test_characterize_dem_isolated_sodium(model, parameters):
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", model,
         "--parameters", json.dumps(parameters)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines()
             if line.startswith("{")]
    assert len(lines) == 3
    assert all(row["isolated_flux"] > 0 and row["predicted_flux"] > 0
               for row in lines)
    if model == "wdem":
        assert all(row["weights_unchanged"] for row in lines)
        assert all(row["nonzero_bins"] == 1 for row in lines)
        for row in lines:
            assert row["isolated_flux"] == pytest.approx(row["predicted_flux"], rel=2.e-8)
        from sherpa.astro import xspec
        from sherpa_plasma import wrap_triplet_pumping

        xspec.set_xsabund("angr")
        xspec.set_xsxset("APECROOT", ROOT)
        child = xspec.XSwdem("pumped_wdem")
        for name, value in parameters.items():
            getattr(child, name).val = value
        wrapper = wrap_triplet_pumping(child, ["Na"])
        edges = np.linspace(1., 1.2, 2001)
        bare = child(edges[:-1], edges[1:])
        np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]), bare)
        wrapper.p_Na = 1.
        actual = wrapper(edges[:-1], edges[1:]) - bare
        record = wrapper._entries[0][0]
        calibration = record["calibration"]
        f, i1, i2 = [row["isolated_flux"] for row in lines]
        transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
        expected = np.zeros_like(actual)
        for row, factor in zip(lines, (-1., *calibration["redistribution"])):
            expected[row["observed_bin"]] += factor * transfer
        np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)
        diagnostic, = wrapper.diagnostics()
        f, i1, i2 = [row["isolated_flux"] for row in lines]
        assert diagnostic["R0_integrated"] == pytest.approx(f / (i1 + i2), rel=2.e-8)
        assert diagnostic["R_integrated"] == pytest.approx(
            (f - transfer) / (i1 + i2 + transfer), rel=2.e-8)
        with patch("sherpa_plasma._uv_model._capture_cie_evaluation",
                   side_effect=AssertionError("unexpected child evaluation")):
            wrapper.p_Na = 2.
            wrapper(edges[:-1], edges[1:])
        diagnostic, = wrapper.diagnostics()
        assert diagnostic["R_integrated"] == pytest.approx(
            diagnostic["R0_integrated"] / 3)
    else:
        assert all(not row["weights_unchanged"] for row in lines)
        assert all(row["nonzero_bins"] > 1 for row in lines)


def test_wdem_rejects_unverified_configurations_and_out_of_coverage():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSwdem("wdem_rejection")
    wrapper = wrap_triplet_pumping(child, ["Na"])
    edges = np.linspace(1., 1.2, 2001)
    wrapper.p_Na = 1.
    with pytest.raises(ValueError, match="incomplete|outside"):
        wrapper(edges[:-1], edges[1:])
    child.Tmax = .8
    child.beta = .5
    child.switch = 1
    with pytest.raises(ValueError, match="switch=2"):
        wrapper(edges[:-1], edges[1:])
    child.switch = 2
    child.nH = 2
    with pytest.raises(ValueError, match="nH=1"):
        wrapper(edges[:-1], edges[1:])


@pytest.mark.parametrize("parameter", ["Na", "norm"])
def test_pumping_checks_temperature_even_at_zero_scale(parameter):
    child, wrapper = _model()
    child.kT = .1
    getattr(child, parameter).val = 0.
    wrapper.p_Na = 1.
    edges = np.linspace(1., 1.2, 2001)
    with pytest.raises(ValueError, match="outside complete triplet coverage"):
        wrapper(edges[:-1], edges[1:])


def test_wdem_trial_parameters_are_used_without_mutating_child():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSwdem("wdem_trial")
    child.Tmax = .8
    child.beta = .5
    wrapper = wrap_triplet_pumping(child, ["Na"])
    wrapper.p_Na = 1.
    edges = np.linspace(1., 1.2, 2001)
    trial = np.array([par.val for par in wrapper.pars])
    trial[wrapper._indices["tmax"]] = .7
    trial[wrapper._indices["beta"]] = .7
    actual = wrapper.calc(trial, edges[:-1], edges[1:])
    assert child.Tmax.val == .8 and child.beta.val == .5
    child.Tmax = .7
    child.beta = .7
    np.testing.assert_array_equal(actual, wrapper(edges[:-1], edges[1:]))


def test_wdem_multiple_ions_and_inactive_coverage():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSwdem("wdem_two_ions")
    child.Tmax = .8
    child.beta = .5
    both = wrap_triplet_pumping(child, ["O", "Na"])
    oxygen = wrap_triplet_pumping(child, ["O"])
    sodium = wrap_triplet_pumping(child, ["Na"])
    edges = np.linspace(.5, 1.2, 4001)
    base = child(edges[:-1], edges[1:])
    both.p_O = oxygen.p_O = 1.
    both.p_Na = sodium.p_Na = 1.
    expected = oxygen(edges[:-1], edges[1:]) + sodium(edges[:-1], edges[1:]) - base
    np.testing.assert_allclose(both(edges[:-1], edges[1:]), expected,
                               rtol=2.e-14, atol=1.e-18)
    assert set(both.temperature_coverage()["ions_keV"]) == {"O", "Na"}
    child.Tmax = .3
    child.beta = .5
    both.p_Na = 0.
    both(edges[:-1], edges[1:])
    both.p_Na = 1.
    with pytest.raises(ValueError, match="Na temperature is outside"):
        both(edges[:-1], edges[1:])


@pytest.mark.parametrize(("model", "element", "abundance"), [
    ("vwdem", "Na", .4),
    ("vvwdem", "Na", .4),
    ("vwdem", "P", None),
    ("vvwdem", "P", .4),
])
def test_variable_wdem_against_isolated_lines(model, element, abundance):
    _require_dem_reference()
    import sherpa_plasma
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping
    from sherpa_plasma._uv_lines import _interpolate_native_lines, _read_native_lines
    from sherpa_plasma.plasma import LineID, _element

    z = _element(element)
    manifest_path = Path(sherpa_plasma.__file__).parent / "_uv_atomdb_3_0_9.json"
    record = json.loads(manifest_path.read_text())["ions"][str(z)]
    calibration = record["calibration"]
    lines = tuple(LineID(z, z - 1, upper, calibration["ground"])
                  for upper in (calibration["metastable"], *calibration["triplet_p"][1:]))
    grid = _read_native_lines(ROOT + "_line.fits", lines)
    temperature = record["reference_temperature_keV"]
    _, wavelengths = _interpolate_native_lines(grid, temperature)
    energies = 12.39841974 / wavelengths / 1.02
    energy_min, energy_max = float(energies.min() * .98), float(energies.max() * 1.02)
    parameters = {"Tmax": temperature * 1.1, "beta": .8, "Redshift": .02}
    if abundance is not None:
        parameters[element] = abundance
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", model,
         "--parameters", json.dumps(parameters), "--element", str(z),
         "--energy-min", str(energy_min), "--energy-max", str(energy_max)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    assert len(rows) == 3
    assert all(row["weights_unchanged"] and row["nonzero_bins"] == 1 for row in rows)
    for row in rows:
        assert row["isolated_flux"] == pytest.approx(row["predicted_flux"], rel=2.e-8)
    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = getattr(xspec, "XS" + model)("variable_" + model + element)
    for name, value in parameters.items():
        getattr(child, name).val = value
    wrapper = wrap_triplet_pumping(child, element)
    edges = np.linspace(energy_min, energy_max, 2001)
    bare = child(edges[:-1], edges[1:])
    np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]), bare)
    getattr(wrapper, "p_" + element).val = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    f, i1, i2 = [row["isolated_flux"] for row in rows]
    transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
    expected = np.zeros_like(actual)
    for row, factor in zip(rows, (-1., *calibration["redistribution"])):
        expected[row["observed_bin"]] += factor * transfer
    np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)
    diagnostic, = wrapper.diagnostics()
    assert diagnostic["R_integrated"] == pytest.approx(
        diagnostic["R0_integrated"] / 2)


@pytest.mark.parametrize(("model", "parameters"), [
    ("bwdem", {"Tmax": .8, "beta": .5, "Redshift": .02, "Velocity": 0.}),
    ("bvwdem", {"Tmax": .8, "beta": .5, "Redshift": .02, "Velocity": 1000.,
                "Na": .6}),
])
def test_characterize_broadened_wdem_lines(model, parameters):
    _require_dem_reference()
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", model,
         "--parameters", json.dumps(parameters)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    assert len(rows) == 3
    for row in rows:
        assert row["weights_unchanged"]
        assert row["nonzero_bins"] > 1
        assert row["isolated_flux"] == pytest.approx(row["predicted_flux"], rel=2.e-8)


@pytest.mark.parametrize(("model", "velocity", "clipped"), [
    ("bwdem", 0., False),
    ("bwdem", 1000., True),
    ("bvwdem", 0., False),
    ("bvwdem", 1000., False),
    ("bvvwdem", 0., False),
    ("bvvwdem", 1000., False),
    ("bvvwdem", 2500., True),
    ("bvvwdem", 2500., "tail"),
])
def test_broadened_wdem_against_isolated_line_profiles(model, velocity, clipped):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    parameters = {"Tmax": .8, "beta": .5, "Redshift": .02,
                  "Velocity": velocity, "norm": .3}
    if model == "bwdem":
        parameters["abundanc"] = .6
    else:
        parameters["Na"] = .6
    if clipped == "tail":
        energy_min, energy_max = 1.075, 1.08
    elif clipped:
        energy_min, energy_max = 1.084, 1.09
    else:
        energy_min, energy_max = 1., 1.2
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", model,
         "--parameters", json.dumps(parameters), "--include-profile",
         "--energy-min", str(energy_min), "--energy-max", str(energy_max)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    assert len(rows) == 3 and all(row["weights_unchanged"] for row in rows)
    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = getattr(xspec, "XS" + model)("profile_" + model)
    for name, value in parameters.items():
        getattr(child, name).val = value
    wrapper = wrap_triplet_pumping(child, "Na")
    edges = np.linspace(energy_min, energy_max, 2001)
    bare = child(edges[:-1], edges[1:])
    np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]), bare)
    wrapper.p_Na = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    f, i1, i2 = [row["predicted_flux"] for row in rows]
    transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
    eta1, eta2 = wrapper._entries[0][0]["calibration"]["redistribution"]
    profiles = [np.asarray(row["isolated_profile"]) for row in rows]
    expected = (-transfer / f * profiles[0] + eta1 * transfer / i1 * profiles[1]
                + eta2 * transfer / i2 * profiles[2])
    np.testing.assert_allclose(actual, expected, rtol=2.e-7, atol=1.e-17)
    if not clipped:
        assert actual.sum() == pytest.approx(0., abs=1.e-17)
    diagnostic, = wrapper.diagnostics()
    assert diagnostic["R_integrated"] == pytest.approx(diagnostic["R0_integrated"] / 2)


@pytest.mark.parametrize(("model", "element", "velocity"), [
    ("bvwdem", "O", 500.),
    ("bvwdem", "P", 0.),
    ("bvvwdem", "P", 1500.),
    ("bwdem", "Fe", 1000.),
])
def test_broadened_wdem_other_ions(model, element, velocity):
    _require_dem_reference()
    import sherpa_plasma
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping
    from sherpa_plasma._uv_lines import _interpolate_native_lines, _read_native_lines
    from sherpa_plasma.plasma import LineID, _element

    z = _element(element)
    manifest_path = Path(sherpa_plasma.__file__).parent / "_uv_atomdb_3_0_9.json"
    record = json.loads(manifest_path.read_text())["ions"][str(z)]
    calibration = record["calibration"]
    lines = tuple(LineID(z, z - 1, upper, calibration["ground"])
                  for upper in (calibration["metastable"], *calibration["triplet_p"][1:]))
    grid = _read_native_lines(ROOT + "_line.fits", lines)
    temperature = record["reference_temperature_keV"]
    _, wavelengths = _interpolate_native_lines(grid, temperature)
    energies = 12.39841974 / wavelengths / 1.02
    energy_min, energy_max = float(energies.min() * .98), float(energies.max() * 1.02)
    parameters = {"Tmax": temperature * 1.1, "beta": .8,
                  "Redshift": .02, "Velocity": velocity}
    if model == "bwdem":
        parameters["abundanc"] = .6
    elif model == "bvvwdem" or element != "P":
        parameters[element] = .6
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", model,
         "--parameters", json.dumps(parameters), "--include-profile",
         "--element", str(z), "--energy-min", str(energy_min),
         "--energy-max", str(energy_max)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    assert len(rows) == 3 and all(row["weights_unchanged"] for row in rows)
    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = getattr(xspec, "XS" + model)("broad_" + element)
    for name, value in parameters.items():
        getattr(child, name).val = value
    wrapper = wrap_triplet_pumping(child, element)
    edges = np.linspace(energy_min, energy_max, 2001)
    bare = child(edges[:-1], edges[1:])
    getattr(wrapper, "p_" + element).val = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    f, i1, i2 = [row["predicted_flux"] for row in rows]
    transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
    eta1, eta2 = calibration["redistribution"]
    profiles = [np.asarray(row["isolated_profile"]) for row in rows]
    expected = (-transfer / f * profiles[0] + eta1 * transfer / i1 * profiles[1]
                + eta2 * transfer / i2 * profiles[2])
    np.testing.assert_allclose(actual, expected, rtol=2.e-7, atol=1.e-17)


def test_broadened_wdem_caching_trial_and_nonuniform_bins():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import _xspec_profiles, wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSbvwdem("broad_cache")
    child.Tmax = .8
    child.beta = .5
    child.Redshift = .02
    child.Velocity = 1000.
    wrapper = wrap_triplet_pumping(child, "Na")
    edges = np.linspace(1., 1.2, 2001)
    wrapper.p_Na = 1.
    result = wrapper(edges[:-1], edges[1:]) - child(edges[:-1], edges[1:])
    with patch.object(_xspec_profiles, "project", side_effect=AssertionError("profile not cached")):
        wrapper.p_Na = 2.
        wrapper(edges[:-1], edges[1:])
    wrapper.p_Na = 1.
    indices = np.unique(np.concatenate((np.arange(0, 1000, 2),
                                        np.arange(1000, 2001, 5), [2000])))
    coarse = edges[indices]
    coarse_result = wrapper(coarse[:-1], coarse[1:]) - child(coarse[:-1], coarse[1:])
    rebinned = np.add.reduceat(result, indices[:-1])
    np.testing.assert_allclose(coarse_result, rebinned, rtol=2.e-7, atol=1.e-17)
    trial = np.array([par.val for par in wrapper.pars])
    trial[wrapper._indices["velocity"]] = 500.
    changed = wrapper.calc(trial, coarse[:-1], coarse[1:])
    assert child.Velocity.val == 1000.
    child.Velocity = 500.
    np.testing.assert_array_equal(changed, wrapper(coarse[:-1], coarse[1:]))


def test_broadened_wdem_multiple_ions_add_independently():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSbvvwdem("broad_two_ions")
    child.Tmax = .8
    child.beta = .5
    child.Redshift = .02
    child.Velocity = 1000.
    both = wrap_triplet_pumping(child, ["O", "Na"])
    oxygen = wrap_triplet_pumping(child, ["O"])
    sodium = wrap_triplet_pumping(child, ["Na"])
    both.p_O = oxygen.p_O = 1.
    both.p_Na = sodium.p_Na = 2.
    edges = np.linspace(.5, 1.2, 4001)
    base = child(edges[:-1], edges[1:])
    expected = oxygen(edges[:-1], edges[1:]) + sodium(edges[:-1], edges[1:]) - base
    np.testing.assert_allclose(both(edges[:-1], edges[1:]), expected,
                               rtol=2.e-14, atol=1.e-18)


@pytest.mark.parametrize("endpoint", ["temperature_min_keV", "temperature_max_keV"])
def test_broadened_profile_temperature_coverage_endpoints(endpoint):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import _xspec_profiles, wrap_triplet_pumping
    from sherpa_plasma._uv_lines import _interpolate_native_lines
    from sherpa_plasma._uv_model import _load_broad_profiles

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSbwdem("broad_endpoint")
    wrapper = wrap_triplet_pumping(child, "Na")
    record, grid, scale = wrapper._entries[0]
    c = record["calibration"]
    profiles = _load_broad_profiles(
        wrapper._library_path, ROOT, 11, c["ground"],
        (c["metastable"], *c["triplet_p"][1:]), record["temperature_min_keV"],
        record["temperature_max_keV"], wrapper._manifest["line_file_sha256"],
        wrapper._manifest["continuum_file_sha256"], wrapper._data_identities)
    temperature = record[endpoint]
    emissivities, _ = _interpolate_native_lines(grid, temperature)
    edges = np.linspace(.9, 1.3, 4001)
    before = _xspec_profiles._snapshot_db()
    actual = np.asarray(_xspec_profiles.project(
        profiles, edges, [temperature], [1.], 0., 1000.)).sum(axis=1)
    assert _xspec_profiles._snapshot_db() == before
    np.testing.assert_allclose(actual, emissivities * 1.e14 * scale,
                               rtol=2.e-8, atol=1.e-18)
    with pytest.raises(RuntimeError, match="invalid DEM weights or temperatures"):
        _xspec_profiles.project(profiles, edges,
                                [record["temperature_max_keV"] * 1.01],
                                [1.], 0., 1000.)


@pytest.mark.parametrize("element", ["O", "Na", "P", "Fe"])
def test_broadened_profile_native_grid_fluxes(element):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import _xspec_profiles, wrap_triplet_pumping
    from sherpa_plasma._uv_model import _load_broad_profiles
    from sherpa_plasma.plasma import _element

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSbwdem("broad_native_" + element)
    wrapper = wrap_triplet_pumping(child, element)
    record, grid, scale = wrapper._entries[0]
    c = record["calibration"]
    profiles = _load_broad_profiles(
        wrapper._library_path, ROOT, _element(element), c["ground"],
        (c["metastable"], *c["triplet_p"][1:]), record["temperature_min_keV"],
        record["temperature_max_keV"], wrapper._manifest["line_file_sha256"],
        wrapper._manifest["continuum_file_sha256"], wrapper._data_identities)
    active = np.flatnonzero((grid.temperatures >= record["temperature_min_keV"])
                            & (grid.temperatures <= record["temperature_max_keV"]))
    wavelength = grid.wavelengths[active[0]]
    energies = 12.39841974 / wavelength
    edges = np.linspace(float(energies.min() * .95),
                        float(energies.max() * 1.05), 1001)
    for index in active:
        actual = np.asarray(_xspec_profiles.project(
            profiles, edges, [grid.temperatures[index]], [1.], 0., 0.)).sum(axis=1)
        expected = grid.emissivities[index] * 1.e14 * scale
        np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-24)


@pytest.mark.parametrize("model", ["XSvwdem", "XSvvwdem"])
def test_variable_wdem_trial_abundance_does_not_mutate_child(model):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = getattr(xspec, model)("variable_trial")
    child.Tmax = .8
    child.beta = .5
    wrapper = wrap_triplet_pumping(child, "Na")
    wrapper.p_Na = 1.
    edges = np.linspace(1., 1.2, 2001)
    trial = np.array([par.val for par in wrapper.pars])
    trial[wrapper._indices["na"]] = .3
    actual = wrapper.calc(trial, edges[:-1], edges[1:])
    assert child.Na.val == 1.
    child.Na = .3
    np.testing.assert_array_equal(actual, wrapper(edges[:-1], edges[1:]))


def test_wdem_oxygen_against_isolated_xspec_lines():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    parameters = {"Tmax": .3, "beta": .5, "Redshift": .02, "abundanc": .6}
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", "wdem",
         "--parameters", json.dumps(parameters), "--element", "8",
         "--energy-min", ".52", "--energy-max", ".59"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines()
             if line.startswith("{")]
    assert len(lines) == 3
    assert all(row["weights_unchanged"] and row["nonzero_bins"] == 1 for row in lines)
    for row in lines:
        assert row["isolated_flux"] == pytest.approx(row["predicted_flux"], rel=2.e-8)
    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSwdem("pumped_wdem_oxygen")
    for name, value in parameters.items():
        getattr(child, name).val = value
    wrapper = wrap_triplet_pumping(child, ["O"])
    edges = np.linspace(.52, .59, 2001)
    bare = child(edges[:-1], edges[1:])
    wrapper.p_O = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    record = wrapper._entries[0][0]
    calibration = record["calibration"]
    f, i1, i2 = [row["isolated_flux"] for row in lines]
    transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
    expected = np.zeros_like(actual)
    for row, factor in zip(lines, (-1., *calibration["redistribution"])):
        expected[row["observed_bin"]] += factor * transfer
    np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)


@pytest.mark.parametrize("element", [
    "C", "N", "O", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Ti", "Cr", "Mn", "Fe", "Ni",
])
def test_wdem_calibrated_ion_against_isolated_lines(element):
    _require_dem_reference()
    import sherpa_plasma
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping
    from sherpa_plasma._uv_lines import _interpolate_native_lines, _read_native_lines
    from sherpa_plasma.plasma import LineID, _element

    z = _element(element)
    manifest_path = Path(sherpa_plasma.__file__).parent / "_uv_atomdb_3_0_9.json"
    record = json.loads(manifest_path.read_text())["ions"][str(z)]
    calibration = record["calibration"]
    lines = tuple(LineID(z, z - 1, upper, calibration["ground"])
                  for upper in (calibration["metastable"], *calibration["triplet_p"][1:]))
    grid = _read_native_lines(ROOT + "_line.fits", lines)
    temperature = record["reference_temperature_keV"]
    _, wavelengths = _interpolate_native_lines(grid, temperature)
    energies = 12.39841974 / wavelengths / 1.02
    energy_min = float(energies.min() * .98)
    energy_max = float(energies.max() * 1.02)
    parameters = {"Tmax": temperature * 1.1, "beta": .8,
                  "Redshift": .02, "abundanc": .6}
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", "wdem",
         "--parameters", json.dumps(parameters), "--element", str(z),
         "--energy-min", str(energy_min), "--energy-max", str(energy_max)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    assert len(rows) == 3
    for row in rows:
        assert row["weights_unchanged"] and row["nonzero_bins"] == 1
        assert row["isolated_flux"] == pytest.approx(row["predicted_flux"], rel=2.e-8)
    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSwdem("wdem_" + element)
    for name, value in parameters.items():
        getattr(child, name).val = value
    wrapper = wrap_triplet_pumping(child, [element])
    edges = np.linspace(energy_min, energy_max, 2001)
    bare = child(edges[:-1], edges[1:])
    getattr(wrapper, "p_" + element).val = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    f, i1, i2 = [row["isolated_flux"] for row in rows]
    transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
    expected = np.zeros_like(actual)
    for row, factor in zip(rows, (-1., *calibration["redistribution"])):
        expected[row["observed_bin"]] += factor * transfer
    np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)


@pytest.mark.parametrize("parameters", [
    {"lowT": .25, "highT": .8, "Redshift": .02, "Abundanc": 1., "norm": .3},
    {"lowT": .3, "highT": .7, "Redshift": .05, "Abundanc": .6, "norm": 1.2},
    {"lowT": .25, "highT": .75, "Redshift": .03, "Abundanc": 1.5, "norm": .5},
])
def test_mkcflow_against_fixed_weight_isolated_lines(parameters):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", "mkcflow",
         "--parameters", json.dumps(parameters), "--fixed-cooling-weights"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines()
             if line.startswith("{")]
    assert len(lines) == 3
    for row in lines:
        assert row["fixed_weight_reference"] and row["nonzero_bins"] == 1
        assert row["isolated_flux"] == pytest.approx(row["predicted_flux"], rel=2.e-8)
    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSmkcflow("pumped_cooling")
    for name, value in parameters.items():
        getattr(child, name).val = value
    wrapper = wrap_triplet_pumping(child, ["Na"])
    edges = np.linspace(1., 1.2, 2001)
    bare = child(edges[:-1], edges[1:])
    np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]), bare)
    wrapper.p_Na = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    record = wrapper._entries[0][0]
    c = record["calibration"]
    f, i1, i2 = [row["isolated_flux"] for row in lines]
    transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
    expected = np.zeros_like(actual)
    for row, factor in zip(lines, (-1., *c["redistribution"])):
        expected[row["observed_bin"]] += factor * transfer
    np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)
    diagnostic, = wrapper.diagnostics()
    f, i1, i2 = [row["isolated_flux"] for row in lines]
    assert diagnostic["R0_integrated"] == pytest.approx(f / (i1 + i2), rel=2.e-8)
    assert diagnostic["R_integrated"] == pytest.approx(
        (f - transfer) / (i1 + i2 + transfer), rel=2.e-8)


@pytest.mark.parametrize("element", [
    "C", "N", "O", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Ti", "Cr", "Mn", "Fe", "Ni",
])
def test_mkcflow_calibrated_ion_against_fixed_weight_lines(element):
    _require_dem_reference()
    import sherpa_plasma
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping
    from sherpa_plasma._uv_lines import _interpolate_native_lines, _read_native_lines
    from sherpa_plasma.plasma import LineID, _element

    z = _element(element)
    manifest_path = Path(sherpa_plasma.__file__).parent / "_uv_atomdb_3_0_9.json"
    record = json.loads(manifest_path.read_text())["ions"][str(z)]
    calibration = record["calibration"]
    lines = tuple(LineID(z, z - 1, upper, calibration["ground"])
                  for upper in (calibration["metastable"], *calibration["triplet_p"][1:]))
    grid = _read_native_lines(ROOT + "_line.fits", lines)
    temperature = record["reference_temperature_keV"]
    _, wavelengths = _interpolate_native_lines(grid, temperature)
    energies = 12.39841974 / wavelengths / 1.02
    parameters = {"lowT": max(.0808, temperature * .9), "highT": temperature * 1.1,
                  "Redshift": .02, "Abundanc": .6}
    script = Path(__file__).parents[1] / "examples" / "characterize_uv_dem.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", "mkcflow",
         "--parameters", json.dumps(parameters), "--element", str(z),
         "--energy-min", str(float(energies.min() * .98)),
         "--energy-max", str(float(energies.max() * 1.02)), "--fixed-cooling-weights"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    assert len(rows) == 3
    for row in rows:
        assert row["fixed_weight_reference"] and row["nonzero_bins"] == 1
        assert row["isolated_flux"] == pytest.approx(row["predicted_flux"], rel=2.e-8)
    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSmkcflow("mkcflow_" + element)
    for name, value in parameters.items():
        getattr(child, name).val = value
    wrapper = wrap_triplet_pumping(child, [element])
    edges = np.linspace(float(energies.min() * .98), float(energies.max() * 1.02), 2001)
    bare = child(edges[:-1], edges[1:])
    getattr(wrapper, "p_" + element).val = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    f, i1, i2 = [row["isolated_flux"] for row in rows]
    transfer = f * (i1 + i2) / (f + 2 * (i1 + i2))
    expected = np.zeros_like(actual)
    for row, factor in zip(rows, (-1., *calibration["redistribution"])):
        expected[row["observed_bin"]] += factor * transfer
    np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)


def test_mkcflow_rejects_unverified_parameters_and_cosmology_changes():
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSmkcflow("cooling_guards")
    child.lowT = .25
    child.highT = .8
    wrapper = wrap_triplet_pumping(child, ["Na"])
    edges = np.linspace(1., 1.2, 2001)
    wrapper.p_Na = 1.
    child.switch = 1
    with pytest.raises(ValueError, match="switch=2"):
        wrapper(edges[:-1], edges[1:])
    child.switch = 2
    cosmology = xspec.get_xscosmo()
    try:
        xspec.set_xscosmo(cosmology[0] + 1., cosmology[1], cosmology[2])
        with pytest.raises(RuntimeError, match="cosmology changed"):
            wrapper(edges[:-1], edges[1:])
    finally:
        xspec.set_xscosmo(*cosmology)


@pytest.mark.parametrize("kind", ["wdem", "vwdem", "vvwdem", "bwdem", "bvwdem", "bvvwdem", "mkcflow"])
def test_zero_pumping_preserves_dem_with_unverified_parameters(kind):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    if kind != "mkcflow":
        child = getattr(xspec, "XS" + kind)("zero_dem")
        child.nH = 2.
    else:
        child = xspec.XSmkcflow("zero_dem")
        child.switch = 1
    wrapper = wrap_triplet_pumping(child, ["Na"])
    edges = np.linspace(1., 1.2, 2001)
    np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]),
                                  child(edges[:-1], edges[1:]))
    wrapper.p_Na = 1.
    with pytest.raises(ValueError, match="nH=1|switch=2"):
        wrapper(edges[:-1], edges[1:])


@pytest.mark.parametrize("kind", ["wdem", "bwdem", "bvwdem", "bvvwdem", "mkcflow"])
def test_dem_pickle_roundtrip_and_cache_lifecycle(kind):
    _require_dem_reference()
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    if kind != "mkcflow":
        child = getattr(xspec, "XS" + kind)("pickle_dem")
        child.Tmax = .8
        child.beta = .5
    else:
        child = xspec.XSmkcflow("pickle_dem")
        child.lowT = .25
        child.highT = .8
        child.Redshift = .02
    wrapper = wrap_triplet_pumping(child, ["Na"])
    wrapper.p_Na = 1.
    edges = np.linspace(1., 1.2, 2001)
    expected = wrapper(edges[:-1], edges[1:])
    restored = pickle.loads(pickle.dumps(wrapper))
    np.testing.assert_array_equal(restored(edges[:-1], edges[1:]), expected)
    assert all(a is b for a, b in zip(restored.pars, restored._child.pars))
    assert restored.p_Na is restored.pars[-1]
    restored.cache_clear()
    assert restored._child_temperatures is None
    assert restored._child_weights is None
    np.testing.assert_array_equal(restored(edges[:-1], edges[1:]), expected)


def test_characterize_zero_velocity_broadened_sodium():
    script = Path(__file__).parents[1] / "examples" / "validate_uv_xspec.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", ROOT, "--model", "bvvapec",
         "--velocity", "0", "--profile", "measure"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines()
             if line.startswith("{")]
    assert len(lines) == 3
    for row in lines:
        assert row["nonzero_bins"] > 1
        assert row["isolated_flux"] == pytest.approx(
            row["linear_native_emissivity"] * 1.e14, rel=2.e-8
        )
        assert row["profile_max_error_relative_total"] is None


def test_zero_pumping_preserves_child_and_parameter_identity():
    child, wrapper = _model()
    edges = np.linspace(1., 1.2, 2001)
    np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]),
                                  child(edges[:-1], edges[1:]))
    assert all(a is b for a, b in zip(wrapper.pars, child.pars))
    assert wrapper.p_Na is wrapper.pars[-1]


def test_pumped_sodium_conserves_isolated_photons(isolated_sodium):
    child, wrapper = _model()
    edges = np.linspace(1., 1.2, 2001)
    bare = child(edges[:-1], edges[1:])
    wrapper.p_Na = 1.
    actual = wrapper(edges[:-1], edges[1:]) - bare
    f, i1, i2 = [row["isolated_flux"] for row in isolated_sodium]
    record = wrapper._entries[0][0]
    c = record["calibration"]
    r0 = f / (i1 + i2)
    target_r = r0 / 2
    delta = f - (f + i1 + i2) * target_r / (1 + target_r)
    expected = np.zeros_like(actual)
    for row, factor in zip(isolated_sodium, (-1, *c["redistribution"])):
        expected[row["observed_bin"]] += factor * delta
    np.testing.assert_allclose(actual, expected, rtol=1.e-10, atol=1.e-18)
    assert actual.sum() == pytest.approx(0., abs=1.e-18)
    assert (f + actual[isolated_sodium[0]["observed_bin"]]) / (
        i1 + i2 + actual[isolated_sodium[1]["observed_bin"]]
        + actual[isolated_sodium[2]["observed_bin"]]
    ) == pytest.approx(target_r, rel=1.e-10)


@pytest.mark.parametrize("table,redshift,abundance,norm,temperature,grid", [
    ("aspl", .1, .4, .02, .44, "linear"),
    ("wilm", .05, 1.7, .3, .46, "quadratic"),
])
def test_pumped_sodium_with_abundances_and_redshift(
        table, redshift, abundance, norm, temperature, grid):
    from sherpa.astro import xspec
    from sherpa_plasma._uv_model import _TripletPumpingModel

    xspec.set_xsxset("APECROOT", ROOT)
    xspec.set_xsabund(table)
    try:
        child = xspec.XSvvapec("scaled_sodium")
        child.kT = temperature
        child.Na = abundance
        child.norm = norm
        child.Redshift = redshift
        wrapper = _TripletPumpingModel(child, ["Na"])
        energy_min, energy_max = .95 / (1 + redshift), 1.22 / (1 + redshift)
        if grid == "linear":
            edges = np.linspace(energy_min, energy_max, 2001)
        else:
            edges = energy_min + (energy_max - energy_min) * np.linspace(0., 1., 2001) ** 2
        script = Path(__file__).parents[1] / "examples" / "validate_uv_xspec.py"
        result = subprocess.run([
            sys.executable, str(script), "--root", ROOT, "--temperature", str(temperature),
            "--redshift", str(redshift), "--abundance", str(abundance),
            "--norm", str(norm), "--abundance-table", table,
            "--energy-min", str(energy_min), "--energy-max", str(energy_max),
            "--grid", grid,
        ], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = [json.loads(line) for line in result.stdout.splitlines()
                 if line.startswith("{")]
        assert len(lines) == 3
        f, i1, i2 = [line["isolated_flux"] for line in lines]
        record = wrapper._entries[0][0]
        calibration = record["calibration"]
        r0 = f / (i1 + i2)
        ratio = r0 / 2
        delta = f - (f + i1 + i2) * ratio / (1 + ratio)
        expected = np.zeros(2000)
        for line, factor in zip(lines, (-1, *calibration["redistribution"])):
            expected[line["observed_bin"]] += factor * delta
        base = child(edges[:-1], edges[1:])
        wrapper.p_Na = 1.
        actual = wrapper(edges[:-1], edges[1:]) - base
        np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)
    finally:
        xspec.set_xsabund("angr")


@pytest.mark.parametrize("model,element,trace", [
    ("apec", "Na", None), ("apec", "Na", "0.7"), ("apec", "Na", "Fe"),
    ("vapec", "Na", None), ("vapec", "Na", "0.7"), ("vapec", "Na", "Fe"),
    ("apec", "O", None), ("vapec", "O", None),
])
def test_apec_variants_against_isolated_xspec(model, element, trace):
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsxset("APECROOT", ROOT)
    xspec.set_xsabund("angr")
    if trace is not None:
        xspec.set_xsxset("APEC_TRACE_ABUND", trace)
    try:
        child = getattr(xspec, "XS" + model)("variant_" + model)
        if model == "apec":
            child.Abundanc = .4
        elif element == "Na":
            child.Fe = .4
        else:
            child.O = .4
        child.kT = .43 if element == "Na" else .2
        wrapper = wrap_triplet_pumping(child, [element])
        energy_min, energy_max = ((1., 1.2) if element == "Na" else (.55, .58))
        edges = np.linspace(energy_min, energy_max, 2001)
        record = wrapper._entries[0][0]
        calibration = record["calibration"]
        upper = (calibration["metastable"], *calibration["triplet_p"][1:])
        script = Path(__file__).parents[1] / "examples" / "validate_uv_xspec.py"
        command = [
            sys.executable, str(script), "--root", ROOT, "--model", model,
            "--element", str(calibration["element"]), "--upper", *map(str, upper),
            "--temperature", str(child.kT.val), "--abundance", ".4",
            "--energy-min", str(energy_min), "--energy-max", str(energy_max),
        ]
        if trace is not None:
            command.extend(("--trace-abundance", trace))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = [json.loads(line) for line in result.stdout.splitlines()
                 if line.startswith("{")]
        assert len(lines) == 3
        f, i1, i2 = [line["isolated_flux"] for line in lines]
        r0 = f / (i1 + i2)
        ratio = r0 / 2
        delta = f - (f + i1 + i2) * ratio / (1 + ratio)
        expected = np.zeros(2000)
        for line, factor in zip(lines, (-1, *calibration["redistribution"])):
            expected[line["observed_bin"]] += factor * delta
        base = child(edges[:-1], edges[1:])
        getattr(wrapper, "p_" + element).val = 1.
        actual = wrapper(edges[:-1], edges[1:]) - base
        np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)
    finally:
        if trace is not None:
            xspec.set_xsxset("APEC_TRACE_ABUND", "")


def test_unsupported_trace_setting_is_rejected():
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsxset("APECROOT", ROOT)
    try:
        xspec.set_xsxset("APEC_TRACE_ABUND", "O")
        with pytest.raises(ValueError, match="nonnegative number or Fe"):
            wrap_triplet_pumping(xspec.XSvapec("bad_trace"), "Na")
        with pytest.raises(ValueError, match="unsupported XSPEC settings"):
            wrap_triplet_pumping(xspec.XSvvapec("vv_trace"), "Na")
    finally:
        xspec.set_xsxset("APEC_TRACE_ABUND", "")


def test_trace_abundance_tracks_trial_iron_parameter():
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsxset("APECROOT", ROOT)
    xspec.set_xsxset("APEC_TRACE_ABUND", "Fe")
    try:
        child = xspec.XSvapec("trace_trial")
        child.kT = .43
        child.Fe = .4
        wrapper = wrap_triplet_pumping(child, "Na")
        wrapper.p_Na = 1.
        edges = np.linspace(1., 1.2, 2001)
        first = wrapper(edges[:-1], edges[1:]) - child(edges[:-1], edges[1:])
        child.Fe = .8
        second = wrapper(edges[:-1], edges[1:]) - child(edges[:-1], edges[1:])
        np.testing.assert_allclose(second, 2 * first, rtol=1.e-10, atol=1.e-18)
    finally:
        xspec.set_xsxset("APEC_TRACE_ABUND", "")


def test_zero_trace_abundance_and_broadened_variant():
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    xspec.set_xsxset("APECROOT", ROOT)
    xspec.set_xsxset("APEC_TRACE_ABUND", "0")
    try:
        child = xspec.XSvapec("zero_trace")
        child.kT = .43
        wrapper = wrap_triplet_pumping(child, "Na")
        wrapper.p_Na = 1.
        edges = np.linspace(1., 1.2, 2001)
        np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]),
                                      child(edges[:-1], edges[1:]))
    finally:
        xspec.set_xsxset("APEC_TRACE_ABUND", "")
    with pytest.raises(TypeError, match="supported XSPEC"):
        wrap_triplet_pumping(xspec.XSbvvapec("broadened"), "Na")


def test_pumping_only_reuses_child_spectrum_and_trial_parameters():
    child, wrapper = _model()
    edges = np.linspace(1., 1.2, 2001)
    parameters = np.array([par.val for par in wrapper.pars])
    with patch.object(type(child), "calc", autospec=True,
                      side_effect=type(child).calc) as calculate:
        original = wrapper.calc(parameters, edges[:-1], edges[1:])
        parameters[-1] = 1.
        pumped = wrapper.calc(parameters, edges[:-1], edges[1:])
        assert calculate.call_count == 1
        assert np.any(original != pumped)
        parameters[0] = .45
        wrapper.calc(parameters, edges[:-1], edges[1:])
        assert calculate.call_count == 2
    assert child.kT.val == pytest.approx(.43)
    wrapper.cache_clear()
    assert wrapper._child_last is None


def test_lifecycle_propagates_to_children_and_invalidates_cache():
    child, wrapper = _model()
    edges = np.linspace(1., 1.2, 2001)
    wrapper(edges[:-1], edges[1:])
    with patch.object(type(child), "startup", autospec=True,
                      side_effect=type(child).startup) as child_start, \
         patch.object(type(child), "teardown", autospec=True,
                      side_effect=type(child).teardown) as child_stop:
        wrapper.startup(cache=True)
        child_start.assert_called_once_with(child, True)
        assert wrapper._child_last is None
        wrapper(edges[:-1], edges[1:])
        wrapper.teardown()
        child_stop.assert_called_once_with(child)
        assert wrapper._child_last is None


def test_changed_apec_file_identity_is_rejected():
    _, wrapper = _model()
    edges = np.linspace(1., 1.2, 2001)
    with patch("sherpa_plasma._uv_model._file_identity", return_value=(0, 0, 0, 0, 0)):
        with pytest.raises(RuntimeError, match="APEC data files changed"):
            wrapper(edges[:-1], edges[1:])
        with pytest.raises(RuntimeError, match="APEC data files changed"):
            wrapper.diagnostics()


def test_unmatched_continuum_is_rejected(tmp_path):
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    root = tmp_path / "mismatched"
    Path(str(root) + "_line.fits").symlink_to(Path(ROOT + "_line.fits"))
    Path(str(root) + "_coco.fits").write_bytes(b"not the verified continuum")
    xspec.set_xsxset("APECROOT", str(root))
    try:
        with pytest.raises(ValueError, match="APEC files"):
            wrap_triplet_pumping(xspec.XSvvapec("mismatched"), "Na")
    finally:
        xspec.set_xsxset("APECROOT", ROOT)


def test_pickle_roundtrip_preserves_child_parameters():
    child, wrapper = _model()
    wrapper.p_Na = 1.
    edges = np.linspace(1., 1.2, 2001)
    expected = wrapper(edges[:-1], edges[1:])
    restored = pickle.loads(pickle.dumps(wrapper))
    np.testing.assert_array_equal(restored(edges[:-1], edges[1:]), expected)
    assert all(a is b for a, b in zip(restored.pars, restored._child.pars))
    assert restored.p_Na is restored.pars[-1]


def test_sherpa_binary_session_roundtrip(tmp_path):
    code = """
import sys
import numpy as np
from sherpa.astro import ui, xspec
from sherpa_plasma import wrap_triplet_pumping
xspec.set_xsxset("APECROOT", sys.argv[1])
child = xspec.XSvvapec("saved_plasma")
child.kT = .43
model = wrap_triplet_pumping(child, "Na")
model.p_Na = 2.
edges = np.linspace(1., 1.2, 2001)
expected = model(edges[:-1], edges[1:])
ui.load_arrays(1, np.arange(1., 11.), np.ones(10))
ui.set_source(1, model)
ui.save(sys.argv[2])
ui.clean()
ui.restore(sys.argv[2])
restored = ui.get_source(1)
assert restored.p_Na.val == 2.
assert restored.pars[0] is restored._child.pars[0]
np.testing.assert_array_equal(restored(edges[:-1], edges[1:]), expected)
"""
    environment = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    result = subprocess.run(
        [sys.executable, "-c", code, ROOT, str(tmp_path / "session.sav")],
        capture_output=True, text=True, check=False, env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("parameter", ["Na", "norm"])
def test_zero_abundance_or_normalization_leaves_child_unchanged(parameter):
    child, wrapper = _model()
    getattr(child, parameter).val = 0.
    wrapper.p_Na = 1.
    edges = np.linspace(1., 1.2, 2001)
    np.testing.assert_array_equal(wrapper(edges[:-1], edges[1:]),
                                  child(edges[:-1], edges[1:]))


def test_multiple_ions_add_independent_corrections():
    from sherpa_plasma._uv_model import _TripletPumpingModel

    child, _ = _model()
    both = _TripletPumpingModel(child, ["O", "Na"])
    oxygen = _TripletPumpingModel(child, ["O"])
    sodium = _TripletPumpingModel(child, ["Na"])
    both.p_O = oxygen.p_O = 1.
    both.p_Na = sodium.p_Na = 2.
    edges = np.linspace(.5, 1.2, 7001)
    base = child(edges[:-1], edges[1:])
    combined = both(edges[:-1], edges[1:]) - base
    independent = (oxygen(edges[:-1], edges[1:]) - base
                   + sodium(edges[:-1], edges[1:]) - base)
    np.testing.assert_allclose(combined, independent, rtol=0, atol=1.e-17)
    coverage = both.temperature_coverage()
    assert coverage["active_intersection_keV"] == (
        max(interval[0] for interval in coverage["ions_keV"].values()),
        min(interval[1] for interval in coverage["ions_keV"].values()),
    )


def test_nonoverlapping_temperature_coverage_is_reported():
    from sherpa_plasma._uv_model import _TripletPumpingModel

    child, _ = _model()
    wrapper = _TripletPumpingModel(child, ["C", "Fe"])
    assert wrapper.temperature_coverage()["active_intersection_keV"] is None
    wrapper.p_C = wrapper.p_Fe = 1.
    assert wrapper.temperature_coverage()["active_intersection_keV"] == ()
    edges = np.linspace(.2, 8., 10001)
    with pytest.raises(ValueError, match="outside complete triplet coverage"):
        wrapper(edges[:-1], edges[1:])


@pytest.mark.parametrize(("element", "forbidden_ev", "intercombination_ev"), [
    ("C", 298.97, 304.41),
    ("N", 419.86, 426.36),
    ("O", 561.02, 568.74),
    ("Ne", 905.00, 915.02),
    ("Mg", 1331.74, 1343.28),
    ("Si", 1839.54, 1853.29),
])
def test_common_line_energies_agree_with_literature(
        element, forbidden_ev, intercombination_ev):
    """Compare rounded line identifications, not atomic pumping rates."""
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping
    from sherpa_plasma._uv_lines import _interpolate_native_lines

    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSvvapec("literature_" + element)
    wrapper = wrap_triplet_pumping(child, element)
    record, grid, _ = wrapper._entries[0]
    _, wavelengths = _interpolate_native_lines(grid, record["reference_temperature_keV"])
    energies = 12398.41974 / wavelengths
    # Table 1: https://arxiv.org/abs/astro-ph/0012218
    assert energies[0] == pytest.approx(forbidden_ev, rel=1.e-3)
    assert energies[1] == pytest.approx(intercombination_ev, rel=1.e-3)


def test_global_setting_changes_are_rejected():
    from sherpa.astro import xspec

    _, wrapper = _model()
    edges = np.linspace(1., 1.2, 2001)
    try:
        xspec.set_xsabund("wilm")
        with pytest.raises(RuntimeError, match="settings or abundances changed"):
            wrapper(edges[:-1], edges[1:])
    finally:
        xspec.set_xsabund("angr")


def test_line_profile_cutoff_setting_is_rejected():
    from sherpa.astro import xspec
    from sherpa_plasma import wrap_triplet_pumping

    child, wrapper = _model()
    edges = np.linspace(1., 1.2, 2001)
    try:
        xspec.set_xsxset("LINECRITLEVEL", "0.001")
        with pytest.raises(RuntimeError, match="settings or abundances changed"):
            wrapper(edges[:-1], edges[1:])
        with pytest.raises(ValueError, match="LINECRITLEVEL"):
            wrap_triplet_pumping(child, "Na")
    finally:
        xspec.set_xsxset("LINECRITLEVEL", "")


def test_synthetic_response_fit_recovers_abundance_and_pumping():
    from sherpa.astro.data import DataARF, DataPHA, DataRMF
    from sherpa.astro.instrument import Response1D
    from sherpa.fit import Fit
    from sherpa.optmethods import LevMar
    from sherpa.stats import LeastSq

    child, wrapper = _model()
    child.kT.freeze()
    child.norm.freeze()
    child.Na.thaw()
    edges = np.linspace(1., 1.2, 2001)
    channels = np.arange(1, len(edges))
    count = len(channels)
    rmf = DataRMF("test_rmf", count, edges[:-1], edges[1:],
                  np.ones(count, dtype=int), channels,
                  np.ones(count, dtype=int), np.ones(count), offset=1,
                  e_min=edges[:-1], e_max=edges[1:])
    arf = DataARF("test_arf", edges[:-1], edges[1:], np.full(count, 100.))
    pha = DataPHA("test_pha", channels, np.zeros(count), exposure=1000.)
    pha.set_response(arf, rmf)
    pha.set_analysis("energy")
    folded = Response1D(pha)(wrapper)
    child.Na = 1.4
    wrapper.p_Na = 2.
    pha.counts = folded(channels)
    child.Na = .7
    wrapper.p_Na = .4
    result = Fit(pha, folded, stat=LeastSq(), method=LevMar()).fit()
    assert result.succeeded
    assert child.Na.val == pytest.approx(1.4, rel=2.e-4)
    assert wrapper.p_Na.val == pytest.approx(2., rel=2.e-4)


def test_off_grid_transfer_and_diagnostics(isolated_sodium):
    child, wrapper = _model()
    wrapper.p_Na = 1.
    edges = np.linspace(1.115, 1.122, 701)
    difference = wrapper(edges[:-1], edges[1:]) - child(edges[:-1], edges[1:])
    assert difference.sum() > 0
    diag, = wrapper.diagnostics()
    assert diag["element"] == "Na"
    assert diag["p"] == 1.
    assert diag["R"] == pytest.approx(diag["R0"] / 2)
    assert len(diag["level_file_sha256"]) == len(diag["radiative_file_sha256"]) == 64
    assert sum(diag["intercombination_redistribution"]) == pytest.approx(1.)


def test_linked_components_keep_distinct_parameters():
    child1, wrapper1 = _model()
    child2, wrapper2 = _model()
    wrapper1.p_Na = 2.
    wrapper2.p_Na.link = wrapper1.p_Na
    assert wrapper1.p_Na is not wrapper2.p_Na
    assert wrapper2.p_Na.val == 2.
    edges = np.linspace(1., 1.2, 2001)
    combined = wrapper1 + wrapper2
    np.testing.assert_allclose(combined(edges[:-1], edges[1:]),
                               wrapper1(edges[:-1], edges[1:])
                               + wrapper2(edges[:-1], edges[1:]))
    assert child1 is not child2


@pytest.mark.parametrize("element", [
    "C", "N", "O", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Ti", "Cr", "Mn", "Fe", "Ni",
])
@pytest.mark.parametrize("temperature_factor", [1., 1.1])
def test_each_candidate_against_isolated_xspec_lines(element, temperature_factor):
    _compare_candidate_with_xspec(element, temperature_factor=temperature_factor)


@pytest.mark.parametrize("element", ["Na", "Mg"])
@pytest.mark.parametrize("endpoint", ["temperature_min_keV", "temperature_max_keV"])
def test_temperature_coverage_endpoints_against_xspec(element, endpoint):
    _compare_candidate_with_xspec(element, endpoint=endpoint)


def _compare_candidate_with_xspec(element, *, temperature_factor=None, endpoint=None):
    from sherpa.astro import xspec
    from sherpa_plasma._uv_lines import _interpolate_native_lines
    from sherpa_plasma._uv_model import _TripletPumpingModel

    xspec.set_xsabund("angr")
    xspec.set_xsxset("APECROOT", ROOT)
    child = xspec.XSvvapec("candidate_" + element)
    wrapper = _TripletPumpingModel(child, [element])
    record, grid, _ = wrapper._entries[0]
    child.kT = (record[endpoint] if endpoint is not None
                else record["reference_temperature_keV"] * temperature_factor)
    _, wavelengths = _interpolate_native_lines(grid, child.kT.val)
    energies = 12.39841974 / wavelengths
    energy_min = float(energies.min() * .98)
    energy_max = float(energies.max() * 1.02)
    edges = np.linspace(energy_min, energy_max, 2001)
    calibration = record["calibration"]
    upper = (calibration["metastable"], *calibration["triplet_p"][1:])
    script = Path(__file__).parents[1] / "examples" / "validate_uv_xspec.py"
    result = subprocess.run([
        sys.executable, str(script), "--root", ROOT,
        "--element", str(calibration["element"]), "--upper", *map(str, upper),
        "--temperature", str(child.kT.val),
        "--energy-min", str(energy_min), "--energy-max", str(energy_max),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines()
             if line.startswith("{")]
    assert len(lines) == 3
    f, i1, i2 = [line["isolated_flux"] for line in lines]
    r0 = f / (i1 + i2)
    ratio = r0 / 2
    delta = f - (f + i1 + i2) * ratio / (1 + ratio)
    expected = np.zeros(2000)
    for line, factor in zip(lines, (-1, *calibration["redistribution"])):
        expected[line["observed_bin"]] += factor * delta
    base = child(edges[:-1], edges[1:])
    getattr(wrapper, "p_" + element).val = 1.
    actual = wrapper(edges[:-1], edges[1:]) - base
    np.testing.assert_allclose(actual, expected, rtol=2.e-8, atol=1.e-18)
    rounding = 4 * np.finfo(float).eps * sum(
        abs(base[line["observed_bin"]]) for line in lines
    )
    assert actual.sum() == pytest.approx(0., abs=rounding + 1.e-18)
