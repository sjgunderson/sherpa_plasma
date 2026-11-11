"""Tests for native AtomDB-backed differential-emission-measure models."""
from collections import OrderedDict
from contextlib import contextmanager
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sherpa_plasma import AtomDBDEMModel, compile_dem_function
from sherpa_plasma.dem_model import _KEV_TO_K, _NativeDEM, _NativeIntegrator


PACKAGE_DIR = Path(__file__).parents[1] / "src" / "sherpa_plasma"
POWERLAW_SOURCE = PACKAGE_DIR / "powerlaw_dem.cpp"
HAS_CPP_COMPILER = shutil.which("c++") is not None or shutil.which("g++") is not None


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
@pytest.mark.parametrize("gamma", [-1.0, -2.0, 0.0])
def test_native_powerlaw_integral_preserves_total_normalization(tmp_path, gamma):
    library = compile_dem_function(POWERLAW_SOURCE, tmp_path / "powerlaw.so")
    dem = _NativeDEM(library=library)
    temperatures = np.array([0.1, 1.0, 10.0])
    spectra = np.tile([2.0, 4.0], (temperatures.size, 1))
    integrator = _NativeIntegrator(
        dem,
        parameters=np.array([gamma]),
        normalization=0.25,
        log_tmin=np.log10(0.11 * _KEV_TO_K),
        log_tmax=np.log10(9.0 * _KEV_TO_K),
        quadrature_points=128,
        mineps_offset=1e-40,
    )

    np.testing.assert_allclose(integrator(temperatures, spectra), [0.5, 1.0])


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_native_integrator_accepts_exact_atomdb_temperature_endpoints(tmp_path):
    low, high = 0.08617333262, 86.17333262
    library = compile_dem_function(POWERLAW_SOURCE, tmp_path / "powerlaw.so")
    integrator = _NativeIntegrator(
        _NativeDEM(library=library),
        parameters=np.array([0.0]),
        normalization=1.0,
        log_tmin=np.log10(low * _KEV_TO_K),
        log_tmax=np.log10(high * _KEV_TO_K),
        quadrature_points=64,
        mineps_offset=1e-40,
    )
    spectra = np.tile([1.0], (3, 1))

    np.testing.assert_allclose(
        integrator(np.array([low, 1.0, high]), spectra), [1.0]
    )


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_native_powerlaw_integrates_temperature_dependent_spectrum(tmp_path):
    library = compile_dem_function(POWERLAW_SOURCE, tmp_path / "powerlaw.so")
    dem = _NativeDEM(library=library)
    temperatures = np.array([0.1, 1.0, 10.0])
    spectra = temperatures[:, None] ** 1.5
    minimum = 0.11 * _KEV_TO_K
    maximum = 9.0 * _KEV_TO_K
    gamma = -1.0
    integrator = _NativeIntegrator(
        dem,
        parameters=np.array([gamma]),
        normalization=2.0,
        log_tmin=np.log10(minimum),
        log_tmax=np.log10(maximum),
        quadrature_points=2048,
        mineps_offset=1e-40,
    )
    expected_mean = (
        (maximum**1.5 - minimum**1.5) / 1.5 / np.log(maximum / minimum)
    ) / _KEV_TO_K**1.5

    actual = integrator(temperatures, spectra)

    np.testing.assert_allclose(actual, [2.0 * expected_mean], rtol=2e-6)


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_custom_dem_source_uses_native_shape_function(tmp_path):
    source = tmp_path / "flat_dem.cpp"
    source.write_text(
        """
        #include "dem_model_abi.h"
        extern "C" SHERPA_PLASMA_DEM_EXPORT int sherpa_plasma_dem_shape_v1(
            double, double, double, size_t n, const double *, double *shape,
            char *, size_t) {
            if (n != 0) return 1;
            *shape = 1.0;
            return 0;
        }
        """
    )
    library = compile_dem_function(source, tmp_path / "flat_dem.so")
    dem = _NativeDEM(library=library)
    integrator = _NativeIntegrator(
        dem,
        parameters=np.array([]),
        normalization=3.0,
        log_tmin=np.log10(0.11 * _KEV_TO_K),
        log_tmax=np.log10(9.0 * _KEV_TO_K),
        quadrature_points=32,
        mineps_offset=1e-40,
    )
    spectra = np.tile([1.5, 2.5], (3, 1))

    np.testing.assert_allclose(
        integrator(np.array([0.1, 1.0, 10.0]), spectra), [4.5, 7.5]
    )


class _FakeSession:
    def __init__(self):
        self.spectra = type("Spectra", (), {"kTlist": [0.1, 1.0, 10.0]})()
        self.calls = 0

    def set_response(self, edges, raw):
        assert raw is True
        self.edges = edges

    def return_spectrum(self, temperature, teunit, nearest):
        assert teunit == "keV"
        assert nearest is True
        self.calls += 1
        return np.array([1.0, 2.0])


class _FakeAtomDB:
    def __init__(self):
        self.session = _FakeSession()

    @property
    def temperature_range(self):
        return 0.1, 10.0

    def _cie(self):
        return self.session

    @contextmanager
    def _using(self):
        yield SimpleNamespace(const=SimpleNamespace(MINEPSOFFSET=1e-40))

    def _configure_session(self, session, abundances, *, abundset=None):
        assert abundances == {}
        assert abundset is None
        return session


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_model_calc_prepares_spectra_once_and_reuses_grid(tmp_path):
    library = compile_dem_function(POWERLAW_SOURCE, tmp_path / "powerlaw.so")
    model = object.__new__(AtomDBDEMModel)
    model._atomdb = _FakeAtomDB()
    model._abundances = {}
    model._abundset = None
    model._quadrature_points = 64
    model._table_cache = OrderedDict()
    model._mineps_offset = 1e-40
    model._native_dem = _NativeDEM(library=library)
    p = np.array([
        2.0,
        np.log10(0.11 * _KEV_TO_K),
        np.log10(9.0 * _KEV_TO_K),
        -1.0,
    ])
    lo = np.array([1.0, 2.0])
    hi = np.array([2.0, 3.0])

    first = model.calc(p, lo, hi)
    second = model.calc(p, lo, hi)

    np.testing.assert_allclose(first, [2.0, 4.0])
    np.testing.assert_allclose(second, first)
    assert model._atomdb.session.calls == 3


def test_model_construction_requires_sherpa():
    from sherpa_plasma import dem_model

    if dem_model._SHERPA_IMPORT_ERROR is None:
        pytest.skip("Sherpa is installed")
    with pytest.raises(ImportError, match="Sherpa is required"):
        AtomDBDEMModel("custom", None)


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_default_model_constructor_registers_powerlaw_parameters(monkeypatch):
    from sherpa_plasma import dem_model

    if dem_model._SHERPA_IMPORT_ERROR is None:
        pytest.skip("uses a minimal Sherpa model test double")

    class FakeParameter:
        def __init__(self, model, name, value, **bounds):
            self.name = name
            self.val = value
            self.bounds = bounds

    def fake_model_init(self, name, parameters):
        self.name = name
        self.pars = parameters

    monkeypatch.setattr(dem_model, "_SHERPA_IMPORT_ERROR", None)
    monkeypatch.setattr(dem_model, "_Parameter", FakeParameter)
    monkeypatch.setattr(dem_model, "AtomDB", _FakeAtomDB)
    monkeypatch.setattr(
        dem_model.ArithmeticModel, "__init__", fake_model_init, raising=False
    )

    model = AtomDBDEMModel("plaw", _FakeAtomDB())

    assert [parameter.name for parameter in model.pars] == [
        "norm", "logTmin", "logTmax", "gamma"
    ]
    assert model.gamma.val == -2.0
    assert model.norm.val == 1e-3
