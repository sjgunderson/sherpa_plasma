"""Module-wide composition handling using controlled XSPEC and AtomDB data."""

import sys
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

import sherpa_plasma
from sherpa_plasma import CapturedPlasmaComponent, LineID, capture_plasma_model, read_plasma_model
from sherpa_plasma._abundances import _abundance_values, _resolve_abundance_spec
from sherpa_plasma.plasma import _SYMBOLS
import test_plasma as plasma_tests
from test_xspec_capture_models import _sherpa_modules


@pytest.fixture
def database():
    case = plasma_tests.PlasmaTests(methodName="runTest")
    case.setUp()
    try:
        yield case
    finally:
        case.doCleanups()


@pytest.fixture
def xspec_state(monkeypatch):
    modules = _sherpa_modules()
    xspec = modules["sherpa.astro.xspec"]
    native = {symbol: 1. for symbol in _SYMBOLS[1:]}
    values = dict(native, O=.25, Ne=1.5, Fe=0.)
    state = {"name": "aspl", "values": values}
    def get_values(table=None):
        if table == "angr":
            return native.copy()
        if table is None or table in ("aspl", "file", "my_table", "lpgp", "lpgs", "felc"):
            return state["values"].copy()
        raise ValueError(f"Unknown abundance table {table!r}")
    xspec.get_xsabund = lambda: state["name"]
    xspec.get_xsabundances = get_values
    xspec.set_xsabund = Mock(side_effect=AssertionError("global abundance state changed"))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state, xspec


@pytest.mark.parametrize("form", ["aspl", "xspec:aspl", "my_table", "lpgp", "lpgs", "felc",
                                  "mapping", "vector", "file"])
def test_find_lines_and_all_emissivity_families_accept_general_compositions(
        database, xspec_state, tmp_path, form):
    state, xspec = xspec_state
    values = state["values"]
    spec = form
    if form == "mapping":
        spec = values
    elif form == "vector":
        spec = [values[symbol] for symbol in _SYMBOLS[1:]]
    elif form == "file":
        spec = tmp_path / "abundances.dat"
        np.savetxt(spec, [values[symbol] for symbol in _SYMBOLS[1:]])
    db = database.db
    line, denominator = LineID(8, 7, 7, 1), LineID(10, 10, 3, 1)
    rows = db.find_lines(21.5, 22.2, element="O", kT=1., abundset=spec)
    np.testing.assert_allclose(rows["Epsilon"], [1.e-15, .5e-15])
    assert db.line_emissivity(line, 1., abundset=spec) == pytest.approx(1.e-15)
    assert db.ratio_emissivity(line, denominator, 1., abundset=spec) == pytest.approx(1 / 3)
    assert db.nei_line_emissivity(line, 1., 1.e11, abundset=spec) == pytest.approx(1.e-15)
    assert db.pshock_line_emissivity(line, 1., 3.e11, tau_lower=1.e11,
                                    abundset=spec) == pytest.approx(2.e-15)
    rows = db.find_nei_lines(21.5, 22.2, temperature=1., ionization_age=1.e11, abundset=spec)
    np.testing.assert_allclose(rows["Epsilon"], [1.e-15, .5e-15])
    rows = db.find_pshock_lines(21.5, 22.2, temperature=1., tau_upper=3.e11,
                              tau_lower=1.e11, abundset=spec)
    np.testing.assert_allclose(rows["Epsilon"], [2.e-15, 1.e-15])
    # Separate fitted multipliers must multiply the chosen baseline, not replace it.
    assert db.line_emissivity(line, 1., abundset=spec,
                              abundances={"O": .4}) == pytest.approx(.4e-15)
    assert db.line_emissivity(line, 1.) == pytest.approx(4.e-15)
    xspec.set_xsabund.assert_not_called()


def test_connection_default_native_and_local_overrides_do_not_leak(database, xspec_state):
    db = database.db
    line = LineID(8, 7, 7, 1)
    db.abundset = "aspl"
    assert db.line_emissivity(line, 1.) == pytest.approx(1.e-15)
    assert db.line_emissivity(line, 1., abundset="native") == pytest.approx(4.e-15)
    assert db.line_emissivity(line, 1., abundset="atomdb:OTHER") == pytest.approx(8.e-15)
    assert db.line_emissivity(line, 1.) == pytest.approx(1.e-15)
    assert db.info["abundance_values"]["O"] == .25


def test_explicit_local_names_remain_local_after_repeated_resolution(xspec_state):
    # DEM construction resolves once; spectrum preparation resolves again.
    _, xspec = xspec_state
    with patch.object(xspec, "get_xsabundances",
                      side_effect=AssertionError("explicit local table resolved through XSPEC")):
        spec = _resolve_abundance_spec("atomdb:aspl")
        assert _resolve_abundance_spec(spec) == "atomdb:aspl"


def test_missing_ag89_reference_file_can_use_xspec_angr(database, xspec_state, monkeypatch):
    db = database.db
    db._cie()
    missing = Mock(side_effect=FileNotFoundError("no local ABUND file"))
    monkeypatch.setattr(db, "_get_data", missing)
    line = LineID(8, 7, 7, 1)
    for _ in range(2):
        assert db.line_emissivity(line, 1., abundset={"O": .25}) == pytest.approx(1.e-15)
    assert missing.call_count == 1  # The reference is cached per session.


def test_positive_abundance_with_zero_native_reference_is_rejected(database, monkeypatch):
    db = database.db
    session = db._cie()
    session._sherpa_plasma_native_abundances = {symbol: 1. for symbol in _SYMBOLS[1:]}
    session._sherpa_plasma_native_abundances["O"] = 0.
    with pytest.raises(ValueError, match="zero reference abundance for O"):
        db.line_emissivity(LineID(8, 7, 7, 1), 1., abundset={"O": .25})


def test_dem_spectral_cache_retains_its_composition_across_other_queries(database, monkeypatch):
    from sherpa_plasma.dem_model import AtomDBDEMModel

    db = database.db
    session = db._cie()
    calls = []
    def response(edges, raw):
        session.energy_edges = edges
    def spectrum(temperature, **kwargs):
        calls.append(temperature)
        return np.full(len(session.energy_edges) - 1,
                       temperature * session.abund[8] * session.abundsetvector[8])
    monkeypatch.setattr(session, "set_response", response, raising=False)
    monkeypatch.setattr(session, "return_spectrum", spectrum, raising=False)
    model = object.__new__(AtomDBDEMModel)
    model._atomdb, model._abundances = db, {"O": 2.}
    model._abundset = _resolve_abundance_spec({"O": .25})
    model._table_cache = OrderedDict()
    edges = np.array([.5, .7])
    _, values = model._prepare_spectra(edges[:-1], edges[1:], edges)
    np.testing.assert_allclose(values[:, 0], .5 * np.array([.1, 1., 10.]))
    db.line_emissivity(LineID(8, 7, 7, 1), 1., abundset="atomdb:other")
    db.abundset = {"O": 9.}
    assert model._prepare_spectra(edges[:-1], edges[1:], edges)[1] is values
    assert len(calls) == 3
    different_edges = np.array([.5, .6, .7])
    _, different = model._prepare_spectra(
        different_edges[:-1], different_edges[1:], different_edges)
    np.testing.assert_allclose(different[:, 0], values[:, 0])
    assert len(calls) == 6


def test_captured_numeric_composition_is_reusable_offline_and_after_global_changes(
        database, xspec_state, monkeypatch):
    state, _ = xspec_state
    state["name"] = "file"
    component = type("XSapec", (), {"parts": (), "__module__": "sherpa.astro.xspec"})()
    component.name = "captured"
    component.pars = [SimpleNamespace(name=name, val=value)
                      for name, value in (("kT", 1.), ("Abundanc", 1.), ("norm", 1.))]
    component.cache_clear = lambda: None
    component.calc = Mock(side_effect=AssertionError("model evaluated during parameter reading"))
    model_state, = read_plasma_model(component)
    component.calc = lambda p, lo, hi: np.ones_like(lo)
    def capture(path, evaluate):
        evaluate()
        return [1.], [1.]
    monkeypatch.setattr(sherpa_plasma, "_xspec_bridge", SimpleNamespace(capture=capture), raising=False)
    snapshot, = capture_plasma_model(component, edges=[.1, 1., 10.])
    assert snapshot.abundance_table == "file"
    state["values"]["O"] = 3.
    assert snapshot.abundance_values["O"] == .25
    assert model_state.abundance_values["O"] == .25
    with pytest.raises(TypeError):
        snapshot.abundance_values["O"] = 9.
    with patch("sherpa_plasma._abundances._xspec_abundance_values",
               side_effect=AssertionError("live XSPEC table read for a detached snapshot")):
        result = database.db.query_plasma_lines(21.5, 22.2, components=[snapshot])
    np.testing.assert_allclose(result.total, [1.e-15, .5e-15])
    with patch("sherpa_plasma.xspec_capture.capture_plasma_model", return_value=(snapshot,)):
        ranked = database.db.find_lines(21.5, 22.2, model=component, brightest=1)
    assert ranked["ModelContribution"][0] == pytest.approx(1.e-15)


@pytest.mark.parametrize("kind", ["xsmkcflow", "xsnei", "xspshock"])
def test_zero_table_elements_are_not_discovered_in_captured_models(database, kind):
    values = {symbol: 1. for symbol in _SYMBOLS[1:]}
    values["O"] = 0.
    parameters = {"Abundanc": 1., "kT": 1., "Tau": 1.e11, "Tau_l": 0., "Tau_u": 1.e11}
    component = CapturedPlasmaComponent(
        "zero", kind, parameters, "file", {},
        [1.] if kind == "xsmkcflow" else None,
        [1.] if kind == "xsmkcflow" else None,
        abundance_values=values)
    result = database.db.query_plasma_lines(21.5, 22.2, components=[component])
    assert len(result.lines) == 0


def test_old_custom_capture_cannot_silently_use_todays_vector(database, xspec_state):
    component = CapturedPlasmaComponent("old", "xsapec", {"Abundanc": 1.}, "file", {}, [1.], [1.])
    with pytest.raises(ValueError, match="captured custom table requires abundance_values"):
        database.db.query_plasma_lines(21.5, 22.2, components=[component])


@pytest.mark.parametrize("values", [{"O": -1.}, {"O": float("nan")},
                                   {"O": [1.]}, {"O": 1., 8: 2.}, [1.] * 29])
def test_invalid_compositions_are_rejected(values):
    with pytest.raises(ValueError):
        _abundance_values(values)


def test_known_table_with_invalid_numbers_does_not_fall_back_to_local_name(xspec_state):
    state, _ = xspec_state
    state["values"]["O"] = float("nan")
    with pytest.raises(ValueError, match="abundance values"):
        _resolve_abundance_spec("aspl")
