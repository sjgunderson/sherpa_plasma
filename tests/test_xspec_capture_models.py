import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

import sherpa_plasma
from sherpa_plasma.xspec_capture import (
    _CIE_TYPES,
    _NEI_TYPES,
    _PSHOCK_TYPES,
    _SUPPORTED_TYPES,
    _capture_cie_evaluation,
    _find_components,
    _single_temperature,
    capture_plasma_model,
)


def _model_class(name):
    return type(name, (), {"parts": ()})


def _sherpa_modules():
    sherpa = ModuleType("sherpa")
    sherpa.__path__ = []
    astro = ModuleType("sherpa.astro")
    astro.__path__ = []
    xspec = ModuleType("sherpa.astro.xspec")
    xspec._xspec = SimpleNamespace(__file__="mock-xspec.so")
    xspec.get_xsabund = lambda: "angr"
    xspec.get_xsxset = lambda: {}
    astro.xspec = xspec
    return {
        "sherpa": sherpa,
        "sherpa.astro": astro,
        "sherpa.astro.xspec": xspec,
    }


class XSpecCaptureModelTests(unittest.TestCase):
    def test_recognizes_atomdb_compatible_cie_nei_and_shock_families(self):
        names = tuple(
            "XS" + model_type[2:]
            for model_type in sorted(_SUPPORTED_TYPES)
        )
        components = tuple(_model_class(name)() for name in names)
        source = type("Compound", (), {"parts": components})()

        found = _find_components(source)
        self.assertEqual(
            [model_type for _, model_type in found],
            [name.lower() for name in names],
        )
        self.assertTrue(all(model_type in _SUPPORTED_TYPES
                            for _, model_type in found))

        self.assertNotIn("xsnlapec", _SUPPORTED_TYPES)
        self.assertTrue(_NEI_TYPES.isdisjoint(_CIE_TYPES))
        self.assertTrue(_PSHOCK_TYPES.isdisjoint(_CIE_TYPES | _NEI_TYPES))
        with self.assertRaisesRegex(ValueError, "Sedov models are not supported"):
            _find_components(_model_class("XSsedov")())

    def test_nei_capture_uses_explicit_parameters_without_evaluation(self):
        component = _model_class("XSvnei")()
        component.name = "nei"
        component.pars = [
            SimpleNamespace(name="kT", val=1.2),
            SimpleNamespace(name="Tau", val=2.e11),
            SimpleNamespace(name="O", val=.5),
        ]
        component.calc = lambda *args: (_ for _ in ()).throw(
            AssertionError("NEI model evaluated during capture")
        )

        with patch.dict("sys.modules", _sherpa_modules()):
            captured, = capture_plasma_model(component)

        self.assertEqual(captured.model_type, "xsvnei")
        self.assertEqual(captured.parameters["Tau"], 2.e11)
        self.assertIsNone(captured.temperatures)
        self.assertIsNone(captured.weights)

    def test_single_temperature_fallback_uses_line_temperature(self):
        temperatures, weights = _single_temperature(
            "xsbapec", {"kT": 2.5}
        )
        assert_array_equal(temperatures, [2.5])
        assert_array_equal(weights, [1.])

        temperatures, weights = _single_temperature(
            "xstapec", {"kT": 4., "kTi": 1.5}
        )
        assert_array_equal(temperatures, [1.5])
        assert_array_equal(weights, [1.])

        self.assertIsNone(_single_temperature("xsgadem", {"Tmean": 2.}))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            _single_temperature("xsapec", {"kT": np.nan})

    def test_capture_uses_scalar_fallback_only_for_single_temperature_models(self):
        parameter = SimpleNamespace(name="kT", val=2.)
        component = _model_class("XSapec")()
        component.name = "apec"
        component.pars = [parameter]
        component.calc = lambda *args: np.zeros(2)
        component.cache_clear = lambda: None
        missing_weights = RuntimeError(
            "Native evaluation did not publish fresh temperature weights"
        )

        def capture_missing(path, evaluate):
            evaluate()
            raise missing_weights

        bridge = SimpleNamespace(capture=capture_missing)
        with patch.dict("sys.modules", _sherpa_modules()), patch.object(
            sherpa_plasma, "_xspec_bridge", bridge, create=True
        ):
            captured, = capture_plasma_model(
                component, edges=[0.1, 1., 10.]
            )
        assert_array_equal(captured.temperatures, [2.])
        assert_array_equal(captured.weights, [1.])

        distribution = _model_class("XSgadem")()
        distribution.name = "gadem"
        distribution.pars = [
            SimpleNamespace(name="Tmean", val=2.),
            SimpleNamespace(name="Tsigma", val=.2),
        ]
        distribution.calc = lambda *args: np.zeros(2)
        distribution.cache_clear = lambda: None
        with patch.dict("sys.modules", _sherpa_modules()), patch.object(
            sherpa_plasma, "_xspec_bridge", bridge, create=True
        ):
            with self.assertRaisesRegex(
                RuntimeError, "no unambiguous single-temperature fallback"
            ):
                capture_plasma_model(distribution, edges=[0.1, 1., 10.])

    def test_trial_evaluation_returns_flux_and_uses_trial_temperature(self):
        component = _model_class("XSapec")()
        component.name = "apec"
        component.pars = [SimpleNamespace(name="kT", val=2.)]
        calls = []
        component.cache_clear = lambda: calls.append("clear")
        component.calc = lambda pars, lo, hi: np.full(lo.shape, pars[0])

        def capture_missing(path, evaluate):
            calls.append("capture")
            evaluate()
            raise RuntimeError("Native evaluation did not publish fresh temperature weights")

        flux, temperatures, weights = _capture_cie_evaluation(
            component, "xsapec", [3.], np.array([1., 2.]), np.array([2., 3.]),
            SimpleNamespace(capture=capture_missing), "mock.so",
        )
        assert_array_equal(flux, [3., 3.])
        assert_array_equal(temperatures, [3.])
        assert_array_equal(weights, [1.])
        self.assertEqual(calls, ["capture", "clear"])
        self.assertEqual(component.pars[0].val, 2.)


if __name__ == "__main__":
    unittest.main()
