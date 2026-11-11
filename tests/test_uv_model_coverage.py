"""Test wrapper coverage plumbing with controlled Sherpa/XSPEC doubles.

These exercise Python correction logic, not native XSPEC model physics.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import sherpa_plasma
from sherpa_plasma._uv_lines import _NativeLineGrid


@pytest.fixture
def uv_module(monkeypatch):
    modules = {name: ModuleType(name) for name in (
        "sherpa", "sherpa.astro", "sherpa.astro.xspec", "sherpa.models",
        "sherpa.models.model", "sherpa.models.parameter")}
    modules["sherpa.astro"].xspec = modules["sherpa.astro.xspec"]
    modules["sherpa.models.model"].ArithmeticModel = type("ArithmeticModel", (), {})
    modules["sherpa.models.model"].CompositeModel = type("CompositeModel", (), {})
    modules["sherpa.models.parameter"].Parameter = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sherpa_plasma, "_xspec_bridge", SimpleNamespace(), raising=False)
    path = Path(sherpa_plasma.__file__).parent / "_uv_model.py"
    spec = importlib.util.spec_from_file_location("sherpa_plasma._uv_model_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_wrapper(module, *, dem=False, broadened=False, temperature=1., norm=1.e-14):
    wrapper = object.__new__(module._TripletPumpingModel)
    wrapper._check_state = lambda: None
    wrapper._validate_dem_trial = lambda trial: None
    wrapper._is_dem, wrapper._is_broadened = dem, broadened
    wrapper._child_count = 4
    wrapper.pars = [SimpleNamespace(val=value) for value in (temperature, norm, 0., 0., 1.)]
    wrapper._indices = {"kt": 0, "norm": 1, "redshift": 2, "velocity": 3}
    wrapper._child = SimpleNamespace(
        pars=wrapper.pars[:4], calc=lambda p, lo, hi: np.full(lo.shape, 10.))
    wrapper._child_last_key = wrapper._child_last = None
    wrapper._child_temperatures = wrapper._child_weights = None
    wrapper._profile_cache = {}
    wrapper._selected = (14,)
    wrapper._abundance_sources = (("constant", 1.),)
    record = {"temperature_min_keV": .2, "temperature_max_keV": 2.,
              "calibration": {"ground": 1, "metastable": 2,
                              "triplet_p": (3, 4, 5), "redistribution": (.75, .25)}}
    grid = _NativeLineGrid(
        np.array([.2, 1., 2.]),
        np.array([[2., 1., 0.], [4., 2., 1.], [1., 0., 0.]]),
        np.tile(12.398419843320026 / np.array([1.1, 1.3, 1.7]), (3, 1)), ())
    wrapper._entries = [(record, grid, 1.)]
    wrapper._manifest = {"line_file_sha256": "line", "continuum_file_sha256": "continuum",
                         "version": "controlled"}
    wrapper._pumping = SimpleNamespace(pars=[SimpleNamespace(val=1.)])
    wrapper._data_identities = ("line", "continuum")
    wrapper._library_path = "controlled-xspec"
    return wrapper


def evaluate(wrapper, *, pumping=1.):
    trial = [par.val for par in wrapper.pars]
    trial[-1] = pumping
    edges = np.array([1., 1.2, 1.5, 1.8])
    return wrapper.calc(trial, edges[:-1], edges[1:])


def test_single_temperature_uses_maskable_arrays(uv_module):
    np.testing.assert_allclose(evaluate(make_wrapper(uv_module)), [8.8, 10.9, 10.3])


@pytest.mark.parametrize("norm", [0., 1.e-14])
def test_single_temperature_outside_former_coverage_accepts_zero_scale(uv_module, norm):
    wrapper = make_wrapper(uv_module, temperature=.1, norm=norm)
    actual = evaluate(wrapper)
    if norm == 0:
        np.testing.assert_array_equal(actual, [10., 10., 10.])
    else:
        np.testing.assert_allclose(actual, [9.5, 10.375, 10.125])


@pytest.mark.parametrize("broadened", [False, True])
def test_dem_integration_and_profiles_share_selection_and_cache(uv_module, monkeypatch, broadened):
    wrapper = make_wrapper(uv_module, dem=True, broadened=broadened)
    captures, projections = [], []

    def capture(child, kind, trial, lo, hi, bridge, library):
        captures.append(trial.copy())
        return np.full(lo.shape, 10.), np.array([.1, 1., 3.]), np.ones(3)

    def project(profiles, edges, temperatures, weights, redshift, velocity):
        projections.append((temperatures.copy(), weights.copy()))
        return np.diag([7., 3., 1.]) * 1.e14

    monkeypatch.setattr(uv_module, "_capture_cie_evaluation", capture)
    monkeypatch.setattr(uv_module, "_load_broad_profiles", lambda *args: "profiles")
    monkeypatch.setattr(uv_module, "_root", lambda: Path("unused"))
    monkeypatch.setattr(sherpa_plasma, "_xspec_profiles", SimpleNamespace(project=project), raising=False)
    transfer = 28. / 15.
    np.testing.assert_allclose(evaluate(wrapper),
                               10. + transfer * np.array([-1., .75, .25]))
    assert len(captures) == 1
    if broadened:
        assert len(projections) == 1
        np.testing.assert_array_equal(projections[0][0], [.1, 1., 3.])
        np.testing.assert_array_equal(projections[0][1], [1., 1., 1.])
    assert np.isfinite(evaluate(wrapper, pumping=2.)).all()
    assert len(captures) == 1
    assert len(projections) == int(broadened)


def test_dem_diagnostics_use_the_full_distribution(uv_module, monkeypatch):
    wrapper = make_wrapper(uv_module, dem=True)
    monkeypatch.setattr(uv_module, "_capture_cie_evaluation", lambda *args: (
        np.ones(3), np.array([.1, 1., 3.]), np.ones(3)))
    diagnostic, = wrapper._dem_diagnostics()
    assert diagnostic["temperature_count"] == 3
    assert diagnostic["temperature_range_keV"] == (.1, 3.)
    assert diagnostic["intrinsic_triplet_fluxes"] == (7., 3., 1.)
    assert diagnostic["R0_integrated"] == 7. / 4.
    assert diagnostic["R_integrated"] == 7. / 8.


@pytest.mark.parametrize("fluxes,status,ratio", [
    ([0., 0., 0.], "no_triplet_emission", None),
    ([1., 0., 0.], "no_intercombination_emission", float("inf")),
    ([0., 1., 0.], "defined", 0.),
])
def test_degenerate_triplets_are_noops_with_explicit_diagnostics(
        uv_module, monkeypatch, fluxes, status, ratio):
    wrapper = make_wrapper(uv_module, dem=True, broadened=True)
    wrapper._entries[0][1].emissivities[:] = fluxes
    monkeypatch.setattr(uv_module, "_capture_cie_evaluation", lambda *args: (
        np.full(3, 10.), np.array([.1, 1., 3.]), np.ones(3)))
    np.testing.assert_array_equal(evaluate(wrapper), [10., 10., 10.])
    diagnostic, = wrapper._dem_diagnostics()
    assert diagnostic["ratio_status"] == status
    assert diagnostic["R_integrated"] == ratio


def test_broadened_transfer_can_populate_an_absent_branch(uv_module, monkeypatch):
    wrapper = make_wrapper(uv_module, dem=True, broadened=True)
    wrapper._entries[0][1].emissivities[:, 2] = 0.
    monkeypatch.setattr(uv_module, "_capture_cie_evaluation", lambda *args: (
        np.full(3, 10.), np.array([.1, 1., 3.]), np.ones(3)))
    monkeypatch.setattr(uv_module, "_load_broad_profiles", lambda *args: "profiles")
    monkeypatch.setattr(uv_module, "_root", lambda: Path("unused"))
    donors = []
    def donor(*args):
        donors.append(args)
        return np.diag([7., 7., 7.]) * 1.e14
    monkeypatch.setattr(sherpa_plasma, "_xspec_profiles", SimpleNamespace(
        project=lambda *args: np.diag([7., 3., 0.]) * 1.e14,
        project_transfer=donor), raising=False)
    transfer = 21. / 13.
    actual = evaluate(wrapper)
    np.testing.assert_allclose(actual, 10. + transfer * np.array([-1., .75, .25]))
    assert actual.sum() == pytest.approx(30.)
    np.testing.assert_array_equal(donors[0][2], [.1, 1., 3.])
    evaluate(wrapper, pumping=2.)
    assert len(donors) == 1


@pytest.mark.parametrize("tolerance", [-1., 1., float("inf"), float("nan")])
def test_public_wrapper_rejects_invalid_coverage_tolerance(uv_module, tolerance):
    with pytest.raises(ValueError, match="coverage_tolerance must be"):
        uv_module.wrap_triplet_pumping(object(), ["Si"], coverage_tolerance=tolerance)
