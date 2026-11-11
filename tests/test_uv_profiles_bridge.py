"""Compile the production C++ bridge against a controlled Aped interface.

These verify native bridge bookkeeping and sparse records. Real XSPEC profile
physics and ABI compatibility are covered separately in test_uv_model_real.py.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import pytest


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler or sys.platform not in ("linux", "darwin"):
        pytest.skip("requires a C++ compiler and dlopen")
    root = Path(__file__).parents[1]
    directory = tmp_path_factory.mktemp("profile_bridge")
    mass = directory / "mass.cpp"
    mass.write_text("double getAtomicMass(const int& element) { return element; }\n")
    output = directory / ("_xspec_profiles" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [compiler, "-std=c++11", "-shared", "-fPIC",
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


@pytest.mark.parametrize("temperatures,weights", [([0.], [1.]), ([1.], [-1.])])
def test_invalid_native_projection_inputs_still_fail(bridge, temperatures, weights):
    with pytest.raises(RuntimeError, match="invalid DEM weights or temperatures"):
        bridge.project(create(bridge), [1., 1.2, 1.8], temperatures, weights, 0., 0.)
