import numpy as np
import pytest

pytest.importorskip("sherpa_plasma._xspec_bridge")

from sherpa.astro import xspec
from sherpa.astro.xspec import _xspec

from sherpa_plasma import _xspec_bridge
from sherpa_plasma.xspec_capture import (
    _CIE_TYPES, _capture_cie_evaluation, capture_plasma_model,
)


@pytest.fixture(
    params=(xspec.XSapec, xspec.XSmkcflow),
    ids=("apec", "mkcflow"),
)
def plasma_model(request):
    model = request.param("capture_test")
    model.Redshift = 0.01
    model.norm = 1.0

    edges = np.geomspace(0.01, 100.0, 1001)
    return model, edges[:-1], edges[1:]


def _evaluate(model, lo, hi):
    parameters = np.asarray([par.val for par in model.pars], dtype=float)
    model.cache_clear()
    return np.asarray(model.calc(parameters, lo, hi), dtype=float)


def _snapshot(path):
    temperatures, weights = _xspec_bridge.snapshot(path)
    return tuple(temperatures), tuple(weights)


def _assert_valid_capture(temperatures, weights):
    temperatures = np.asarray(temperatures, dtype=float)
    weights = np.asarray(weights, dtype=float)

    assert temperatures.ndim == 1
    assert weights.ndim == 1
    assert temperatures.size > 0
    assert temperatures.size == weights.size
    assert np.isfinite(temperatures).all()
    assert np.all(temperatures > 0)
    assert np.isfinite(weights).all()
    assert np.any(weights != 0)


def test_capture_returns_fresh_metadata_and_restores_previous_state(
    plasma_model,
):
    model, lo, hi = plasma_model
    library_path = _xspec.__file__

    flux = _evaluate(model, lo, hi)
    assert flux.shape == lo.shape
    assert np.isfinite(flux).all()

    before = _snapshot(library_path)
    assert before[0]
    assert before[1]

    temperatures, weights = _xspec_bridge.capture(
        library_path,
        lambda: _evaluate(model, lo, hi),
    )

    _assert_valid_capture(temperatures, weights)
    assert _snapshot(library_path) == before


def test_capture_restores_metadata_if_evaluation_raises(plasma_model):
    model, lo, hi = plasma_model
    library_path = _xspec.__file__

    flux = _evaluate(model, lo, hi)
    assert np.isfinite(flux).all()
    before = _snapshot(library_path)

    def evaluate_then_raise():
        _evaluate(model, lo, hi)
        raise RuntimeError("intentional callback failure")

    with pytest.raises(RuntimeError, match="intentional callback failure"):
        _xspec_bridge.capture(library_path, evaluate_then_raise)

    assert _snapshot(library_path) == before


def test_capture_rejects_callback_that_does_not_evaluate_xspec(plasma_model):
    model, lo, hi = plasma_model
    library_path = _xspec.__file__

    flux = _evaluate(model, lo, hi)
    assert np.isfinite(flux).all()
    before = _snapshot(library_path)

    with pytest.raises(
        RuntimeError,
        match="did not publish fresh temperature weights",
    ):
        _xspec_bridge.capture(library_path, lambda: None)

    assert _snapshot(library_path) == before


@pytest.mark.parametrize("model_class", [xspec.XSwdem, xspec.XSmkcflow])
def test_native_dem_weights_reconstruct_apec_spectrum(model_class):
    model = model_class("dem_reconstruction")
    model.Redshift = .1
    edges = np.linspace(.9, 1.3, 1001)
    lo, hi = edges[:-1], edges[1:]
    trial = np.array([par.val for par in model.pars], dtype=float)
    trial[-1] = .3
    flux, temperatures, weights = _capture_cie_evaluation(
        model, type(model).__name__.lower(), trial, lo, hi,
        _xspec_bridge, _xspec.__file__,
    )
    apec = xspec.XSapec("dem_reference")
    apec.Redshift = model.Redshift.val
    apec.norm = .3
    reconstructed = np.zeros_like(flux)
    for temperature, weight in zip(temperatures, weights):
        apec.kT = temperature
        reconstructed += weight * apec(lo, hi)
    np.testing.assert_allclose(flux, reconstructed, rtol=2.e-11, atol=1.e-18)
    assert model.norm.val == 1.


@pytest.mark.parametrize("model_type", sorted(_CIE_TYPES))
def test_capture_supported_cie_model_families(model_type):
    class_name = "XS" + model_type[2:]
    model_class = getattr(xspec, class_name, None)
    if model_class is None:
        pytest.skip(f"{class_name} is unavailable in this Sherpa/XSPEC build")

    model = model_class("capture_matrix")
    captured, = capture_plasma_model(model, edges=[0.1, 1.0, 10.0])

    assert captured.model_type == model_type
    _assert_valid_capture(captured.temperatures, captured.weights)
