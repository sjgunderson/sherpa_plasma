"""Abundance-table handling with controlled XSPEC state and model doubles."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import sherpa_plasma
from test_uv_model_coverage import uv_module, make_wrapper, evaluate


@pytest.fixture
def composition(uv_module, monkeypatch):
    state = {"table": "file", "values": {"Si": .005, "He": .1}}
    reference = {"Si": .002, "He": .1}
    monkeypatch.setattr(uv_module.xspec, "get_xsversion", lambda: "12.14.0k", raising=False)
    monkeypatch.setattr(uv_module.xspec, "get_xsxset", lambda: {}, raising=False)
    monkeypatch.setattr(uv_module.xspec, "get_xsabund", lambda: state["table"], raising=False)
    monkeypatch.setattr(uv_module.xspec, "get_xsabundances",
                        lambda table=None: dict(reference if table == "angr" else state["values"]),
                        raising=False)
    return state, reference


@pytest.mark.parametrize("table", ["angr", "aneb", "aspl", "felc", "feld", "grsa",
                                   "lodd", "lpgp", "lpgs", "wilm", "file", "new_table"])
def test_scaling_uses_values_without_a_table_name_whitelist(uv_module, composition, table):
    state, reference = composition
    state["table"] = table
    if table == "angr":
        state["values"] = reference.copy()
    assert uv_module._abundance_scales([14]) == {14: 1. if table == "angr" else 2.5}
    assert state["table"] == table


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_selected_abundance_is_reported(uv_module, composition, value):
    state, _ = composition
    state["values"]["Si"] = value
    with pytest.raises(ValueError, match="Si XSPEC abundance must be"):
        uv_module._abundance_scales([14])


def test_zero_abundance_is_supported(uv_module, composition):
    state, _ = composition
    state["values"]["Si"] = 0.
    assert uv_module._abundance_scales([14]) == {14: 0.}


@pytest.mark.parametrize("value", [0., -1., float("nan")])
def test_reference_abundance_must_be_positive(uv_module, composition, value):
    _, reference = composition
    reference["Si"] = value
    with pytest.raises(ValueError, match="positive reference XSPEC abundance"):
        uv_module._abundance_scales([14])


@pytest.mark.parametrize("element", ["Si", "He"])
def test_custom_vector_reload_with_same_name_invalidates_wrapper(
        uv_module, composition, element):
    state, _ = composition
    wrapper = SimpleNamespace(_state_at_init=uv_module._state(), _data_files=(),
                              _data_identities=(), _is_mkcflow=False)
    uv_module._TripletPumpingModel._check_state(wrapper)
    state["values"][element] *= 2.
    assert state["table"] == "file"
    with pytest.raises(RuntimeError, match="settings or abundances changed"):
        uv_module._TripletPumpingModel._check_state(wrapper)


def test_named_table_state_check_does_not_read_thirty_values(
        uv_module, composition, monkeypatch):
    state, _ = composition
    state["table"] = "lodd"
    def unexpected(*args):
        raise AssertionError("named-table state check read the abundance vector")
    monkeypatch.setattr(uv_module.xspec, "get_xsabundances", unexpected)
    assert uv_module._state()[1] == "lodd"


@pytest.mark.parametrize("kind", ["XSapec", "XSvapec", "XSvvapec", "XSwdem", "XSvwdem",
                                  "XSvvwdem", "XSbwdem", "XSbvwdem", "XSbvvwdem", "XSmkcflow"])
@pytest.mark.parametrize("table", ["lodd", "file"])
def test_all_wrapper_families_construct_with_other_tables(
        uv_module, composition, monkeypatch, tmp_path, kind, table):
    state, _ = composition
    state["table"] = table
    for name in ("XSapec", "XSvapec", "XSvvapec", "XSwdem", "XSvwdem", "XSvvwdem",
                 "XSbwdem", "XSbvwdem", "XSbvvwdem", "XSmkcflow"):
        monkeypatch.setattr(uv_module.xspec, name, type(name, (), {}), raising=False)
    monkeypatch.setattr(uv_module.xspec, "_xspec", SimpleNamespace(__file__="controlled.so"),
                        raising=False)
    monkeypatch.setattr(uv_module.xspec, "get_xscosmo", lambda: (70., 0., .73), raising=False)
    monkeypatch.setattr(sherpa_plasma, "_xspec_profiles",
                        SimpleNamespace(project_transfer=lambda *args: None), raising=False)
    class Parameter:
        def __init__(self, model, name, value, **kwargs):
            self.name, self.val = name, value
    def arithmetic_init(self, name, parameters):
        self.name, self.pars = name, tuple(parameters)
    def composite_init(self, name, parts):
        self.name, self.pars = name, tuple(par for part in parts for par in part.pars)
    monkeypatch.setattr(uv_module, "Parameter", Parameter)
    monkeypatch.setattr(uv_module.ArithmeticModel, "__init__", arithmetic_init)
    monkeypatch.setattr(uv_module.CompositeModel, "__init__", composite_init)
    record, grid, _ = make_wrapper(uv_module)._entries[0]
    manifest = {"line_file_sha256": "line", "continuum_file_sha256": "continuum",
                "ions": {"14": record}}
    filename = tmp_path / "calibration.json"
    filename.write_text(json.dumps(manifest))
    for suffix in ("_line.fits", "_coco.fits"):
        (tmp_path / ("apec" + suffix)).write_bytes(b"controlled data")
    monkeypatch.setattr(uv_module, "_MANIFESTS", (str(filename),))
    monkeypatch.setattr(uv_module, "_root", lambda: tmp_path / "apec")
    monkeypatch.setattr(uv_module, "_file_sha256", lambda path, identity:
                        "line" if str(path).endswith("_line.fits") else "continuum")
    monkeypatch.setattr(uv_module, "_load_grid", lambda *args: grid)
    child = getattr(uv_module.xspec, kind)()
    child.name = "component"
    child.pars = [SimpleNamespace(name=name, val=value) for name, value in
                  (("kT", 1.), ("norm", 1.e-14), ("Redshift", .02), ("Si", .6),
                   ("Abundanc", .6), ("switch", 2.), ("nH", 1.))]
    cached = {"flux": np.full(3, 99.)}  # Left over from another composition.
    child.cache_clear = cached.clear
    def calculate(p, lo, hi):
        if "flux" not in cached:
            cached["flux"] = np.full(lo.shape, state["values"]["Si"] / .002)
        return cached["flux"]
    child.calc = calculate
    wrapper = uv_module.wrap_triplet_pumping(child, ["Si"])
    assert wrapper._entries[0][2] == 2.5
    assert wrapper.p_Si.val == 0.
    edges = np.array([1., 1.2, 1.5, 1.8])
    actual = wrapper.calc([par.val for par in wrapper.pars], edges[:-1], edges[1:])
    np.testing.assert_array_equal(actual, np.full(3, 2.5))


@pytest.mark.parametrize("scale", [0., .25, 2.5])
def test_unbroadened_correction_and_diagnostics_follow_active_composition(
        uv_module, monkeypatch, scale):
    wrapper = make_wrapper(uv_module, dem=True)
    record, grid, _ = wrapper._entries[0]
    wrapper._entries[0] = record, grid, scale
    monkeypatch.setattr(uv_module, "_capture_cie_evaluation", lambda *args: (
        np.full(3, 10.), np.array([1.]), np.ones(1)))
    np.testing.assert_allclose(evaluate(wrapper), 10. + scale * np.array([-1.2, .9, .3]))
    diagnostic, = wrapper._dem_diagnostics()
    np.testing.assert_allclose(diagnostic["intrinsic_triplet_fluxes"],
                               scale * np.array([4., 2., 1.]))
    assert diagnostic["abundance_scale_from_angr"] == scale
    if scale == 0:
        assert diagnostic["ratio_status"] == "no_triplet_emission"
        assert diagnostic["R_integrated"] is None
    else:
        assert diagnostic["R_integrated"] == pytest.approx(2. / 3.)
