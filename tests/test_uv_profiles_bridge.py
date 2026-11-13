"""Compile the production C++ bridge against a controlled Aped interface.

These verify native bridge bookkeeping and sparse records. Real XSPEC profile
physics and ABI compatibility are covered separately in test_uv_model_real.py.
"""
import importlib.util
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import pytest
import sherpa_plasma
from test_uv_model_coverage import uv_module, make_wrapper, evaluate


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler or sys.platform not in ("linux", "darwin"):
        pytest.skip("requires a C++ compiler and dlopen")
    root = Path(__file__).parents[1]
    directory = tmp_path_factory.mktemp("profile_bridge")
    mass = directory / "mass.cpp"
    mass.write_text('''#include <XSFunctions/Utilities/FunctionUtility.h>
double getAtomicMass(const int& element) { return element; }
extern "C" void profile_test_xset(const char* key, const char* value) {
    if (value) FunctionUtility::modelStrings()[key] = value;
    else FunctionUtility::modelStrings().erase(key);
}
''')
    output = directory / ("_xspec_profiles" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [compiler, "-std=c++11", "-shared", "-fPIC", "-pthread",
               "-I" + sysconfig.get_path("include"),
               "-I" + str(root / "tests" / "xspec_profile_stub"),
               str(root / "src" / "sherpa_plasma" / "_xspec_profiles.cpp"),
               str(mass), "-o", str(output)]
    if sys.platform == "darwin":
        command.extend(["-undefined", "dynamic_lookup"])
    else:
        command.extend(["-ldl", "-Wl,-Bsymbolic"])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    spec = importlib.util.spec_from_file_location("_xspec_profiles", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create(bridge, mode="sparse"):
    return bridge.create(bridge.__file__, mode, 14, 1, (2, 5, 6), 1., 1.,
                         tuple(12.39841974 / np.array([1.1, 1.3, 1.7])))


@pytest.fixture
def native_settings(bridge):
    library = ctypes.CDLL(bridge.__file__)
    setter = library.profile_test_xset
    setter.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
    setter.restype = None
    def set_options(**settings):
        for key, value in settings.items():
            setter(key.encode(), None if value is None else value.encode())
    yield set_options
    set_options(APECTHERMAL=None, APECMULTITHREAD=None)


@pytest.mark.parametrize("thermal_default", [False, True])
def test_native_options_follow_xset_overrides(bridge, native_settings, thermal_default):
    profiles = create(bridge)
    assert bridge._profile_options_version == 1
    assert bridge._control_options(profiles, 250., thermal_default) == (thermal_default, False, 250.)
    for value, enabled in (("yes", True), ("no", False), ("ON", True), ("off", False)):
        native_settings(APECTHERMAL=value, APECMULTITHREAD=value)
        assert bridge._control_options(profiles, 250., thermal_default) == (enabled, enabled, 250.)


def test_thermal_profiles_use_each_native_temperature(bridge, native_settings):
    profiles = create(bridge)
    edges = np.linspace(1.08, 1.12, 16001)
    centers = (edges[:-1] + edges[1:]) / 2
    native_settings(APECTHERMAL="yes")
    variance = []
    for temperature in (.1, 1., 10.):
        line = np.asarray(bridge.project(profiles, edges, [temperature], [1.], 0., 0., False))[0]
        variance.append(np.sum(line * (centers - 1.1)**2) / line.sum())
    np.testing.assert_allclose(np.array(variance) / variance[1], [.1, 1., 10.], rtol=2.e-4)
    # A broad b-family default can be switched off explicitly.
    native_settings(APECTHERMAL="no")
    narrow = np.asarray(bridge.project(profiles, edges, [1.], [1.], 0., 0., True))[0]
    assert np.count_nonzero(narrow) == 1


@pytest.mark.parametrize("projector", ["project", "project_transfer"])
def test_multithreaded_projection_matches_serial_and_preserves_diagnostics(
        bridge, native_settings, projector):
    profiles = create(bridge, "absent_i2")
    edges = np.linspace(1.05, 1.75, 4001)
    args = (profiles, edges, [.01, .1, .5, 1., 10., 100.], [1., 2., 3., 4., 5., 6.], .02, 500.)
    native_settings(APECTHERMAL="yes", APECMULTITHREAD="no")
    before = bridge._snapshot_db()
    serial = np.asarray(getattr(bridge, projector)(*args))
    native_settings(APECMULTITHREAD="yes")
    threaded = np.asarray(getattr(bridge, projector)(*args))
    np.testing.assert_array_equal(threaded, serial)
    assert bridge._snapshot_db() == before
    # Scaling a line's emissivity before the linear native projection gives the
    # same answer as scaling its cached projected profile during a pumping step.
    scaled = np.asarray(getattr(bridge, projector)(
        profiles, edges, args[2], np.array(args[3]) * .37, .02, 500.))
    np.testing.assert_allclose(scaled, .37 * threaded, rtol=1.e-14, atol=1.e-10)


def test_sparse_native_profiles_include_both_tails_and_clamp_endpoints(bridge):
    profiles = create(bridge)
    temperatures = [.01, .1, .5, 1., 10., 100.]
    actual = np.asarray(bridge.project(profiles, [1., 1.2, 1.5, 1.8],
                                      temperatures, np.ones(6), 0., 0.))
    expected = [sum(np.interp(temperatures, [.1, 1., 10.], values))
                for values in ([2., 4., 1.], [0., 2., 0.], [0., 1., 0.])]
    np.testing.assert_allclose(actual, np.diag(expected) * 1.e14, rtol=1.e-14)


def test_unlisted_branch_can_receive_forbidden_donor_photons(bridge):
    profiles = create(bridge, "absent_i2")
    args = (profiles, [1., 1.2, 1.5, 1.8], [.1, 1., 10.], [1., 1., 1.], 0., 0.)
    native = np.asarray(bridge.project(*args))
    np.testing.assert_allclose(native, np.diag([7., 2., 0.]) * 1.e14)
    donor = np.asarray(bridge.project_transfer(*args))
    np.testing.assert_allclose(donor, np.eye(3) * 7.e14)
    # Each projection resets mutable emissivities after donor mode was used.
    np.testing.assert_array_equal(bridge.project(*args), native)


def test_transition_ambiguity_still_fails(bridge):
    with pytest.raises(RuntimeError, match="ambiguous"):
        create(bridge, "duplicate")


def test_transition_identity_survives_native_energy_sorting(bridge):
    profiles = create(bridge, "swapped")
    actual = np.asarray(bridge.project(
        profiles, [1., 1.2, 1.5, 1.8], [.1, 1., 10.], [1., 1., 1.], 0., 0.))
    expected = np.array([[7., 0., 0.], [0., 0., 2.], [0., 1., 0.]]) * 1.e14
    np.testing.assert_allclose(actual, expected, rtol=1.e-14)


@pytest.mark.parametrize("mode,scale", [("solar_zero", 0.), ("solar_low", .25),
                                        ("solar_high", 2.5)])
@pytest.mark.parametrize("missing_branch", [False, True])
def test_wrapper_applies_native_solar_conversion_once(
        bridge, uv_module, monkeypatch, mode, scale, missing_branch):
    wrapper = make_wrapper(uv_module, dem=True, broadened=True)
    record, grid, _ = wrapper._entries[0]
    wrapper._entries[0] = record, grid, scale
    if missing_branch:
        grid.emissivities[:, 2] = 0.
        mode += "_absent_i2"
    context = create(bridge, mode)
    monkeypatch.setattr(uv_module, "_capture_cie_evaluation", lambda *args: (
        np.full(3, 10.), np.array([1.]), np.ones(1)))
    monkeypatch.setattr(uv_module, "_load_broad_profiles", lambda *args: context)
    monkeypatch.setattr(uv_module, "_root", lambda: Path("controlled"))
    monkeypatch.setattr(sherpa_plasma, "_xspec_profiles", bridge, raising=False)
    transfer = 1. if missing_branch else 1.2
    np.testing.assert_allclose(evaluate(wrapper),
                               10. + scale * transfer * np.array([-1., .75, .25]))
    # Pumping-only changes continue to use the native profiles already cached.
    def unexpected(*args):
        raise AssertionError("unexpected projection for unchanged child")
    monkeypatch.setattr(uv_module, "_load_broad_profiles", unexpected)
    assert np.isfinite(evaluate(wrapper, pumping=2.)).all()


@pytest.mark.parametrize("temperatures,weights", [([0.], [1.]), ([1.], [-1.])])
def test_invalid_native_projection_inputs_still_fail(bridge, temperatures, weights):
    with pytest.raises(RuntimeError, match="invalid DEM weights or temperatures"):
        bridge.project(create(bridge), [1., 1.2, 1.8], temperatures, weights, 0., 0.)
