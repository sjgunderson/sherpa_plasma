"""Parameter extraction must never evaluate or alter a source expression."""
import importlib.util
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import sherpa_plasma
from sherpa_plasma import AtomDB, read_plasma_model


class Parameter:
    def __init__(self, name, value):
        self.name, self.value = name, value

    @property
    def val(self):
        return self.value() if callable(self.value) else self.value


def component(kind="XSapec", name="hot", **values):
    values = {"kT": .7, "Abundanc": .5, "Redshift": .02, "norm": .001, **values}
    cls = type(kind, (), {})
    part = cls()
    part.name, part.pars = name, [Parameter(k, v) for k, v in values.items()]
    part.calc = Mock(side_effect=AssertionError("model was evaluated"))
    return part


class PlasmaStateTests(unittest.TestCase):
    def test_component_snapshot_is_detached_and_linked_values_are_current(self):
        target = [.7]
        model = component(kT=lambda: target[0])
        state, = read_plasma_model(model)
        self.assertEqual((state.name, state.kT, state.redshift, state.norm), ("hot", .7, .02, .001))
        self.assertEqual(state.metal_abundance, .5)
        self.assertEqual(state.abundances, {})
        with self.assertRaises(TypeError):
            state.parameters["kT"] = 3.
        target[0] = 1.2
        self.assertEqual(state.kT, .7)
        self.assertEqual(read_plasma_model(model)[0].kT, 1.2)
        model.calc.assert_not_called()

    def test_compound_and_shared_wrappers_deduplicate_by_identity(self):
        hot, cool = component(), component(name="cool", kT=.2)
        absorption = SimpleNamespace(parts=())
        expression = SimpleNamespace(parts=(absorption, SimpleNamespace(parts=(hot, cool, hot))))
        wrapper = SimpleNamespace(parts=(expression,))
        self.assertEqual([s.name for s in read_plasma_model(wrapper)], ["hot", "cool"])

    def test_all_supported_variants_and_explicit_abundances(self):
        for kind in ("XSapec", "XSbapec", "XSvapec", "XSbvapec", "XSvvapec", "XSbvvapec"):
            with self.subTest(kind=kind):
                state, = read_plasma_model(component(kind, O=.2, Fe=1.3, Velocity=120.))
                self.assertEqual(state.model_type, kind.lower())
                self.assertEqual(state.abundances, {"O": .2, "Fe": 1.3})
                self.assertEqual(state.velocity, 120.)

    def test_session_lookup_uses_requested_or_default_dataset(self):
        session = SimpleNamespace(get_source=Mock(return_value=component()))
        self.assertEqual(read_plasma_model(dataset_id="meg", session=session)[0].kT, .7)
        session.get_source.assert_called_once_with("meg")
        session.get_source.reset_mock()
        read_plasma_model(session=session)
        session.get_source.assert_called_once_with()
        with patch.dict(sys.modules, {"sherpa.astro": SimpleNamespace(ui=session)}):
            read_plasma_model(dataset_id=2)
        session.get_source.assert_called_with(2)

    def test_xspec_global_settings_are_copied_without_changes(self):
        model = component()
        type(model).__module__ = "sherpa.astro.xspec"
        settings = {"APECROOT": "/local/apec", "APEC_TRACE_ABUND": "1.0"}
        xspec = SimpleNamespace(get_xsabund=Mock(return_value="angr"),
                                get_xsabundances=Mock(return_value={"O": .001}),
                                get_xsxset=Mock(return_value=settings))
        with patch.dict(sys.modules, {"sherpa.astro": SimpleNamespace(xspec=xspec)}):
            state, = read_plasma_model(model)
        self.assertEqual(state.abundance_table, "angr")
        settings["APECROOT"] = "changed"
        self.assertEqual(state.xset["APECROOT"], "/local/apec")
        with self.assertRaises(TypeError):
            state.xset["APECROOT"] = "changed"

    def test_reading_via_database_needs_no_database_files_or_pyatomdb(self):
        with tempfile.TemporaryDirectory() as root, AtomDB(root) as db:
            self.assertEqual(db.read_model(component())[0].kT, .7)
            self.assertIsNone(db._api)
            self.assertIsNone(db._session)

    def test_invalid_and_unsupported_models_fail_explicitly(self):
        for model in (SimpleNamespace(), component("XSpowerlaw"), component(kT=0),
                      component(kT=float("nan")), component(Redshift=-1)):
            with self.assertRaises(ValueError):
                read_plasma_model(model)
        with self.assertRaises(ValueError):
            read_plasma_model(component(), dataset_id=1)

    def test_database_only_public_api(self):
        for name in ("AtomDBModel", "PlawDEM2", "DEMModel", "PlasmaKernel", "PowerLawDEM"):
            self.assertNotIn(name, sherpa_plasma.__all__)
            self.assertFalse(hasattr(sherpa_plasma, name))
        for name in ("spectrum", "create_model", "create_dem_model", "create_plaw_dem2"):
            self.assertFalse(hasattr(AtomDB, name))


HAS_XSPEC = importlib.util.find_spec("sherpa") is not None
if HAS_XSPEC:
    try:
        from sherpa.astro import xspec
    except ImportError:
        HAS_XSPEC = False


@unittest.skipUnless(HAS_XSPEC, "requires Sherpa with the XSPEC extension")
class RealXSPECStateTests(unittest.TestCase):
    def test_compound_expression_and_linked_temperature(self):
        from sherpa.astro import xspec
        hot, cool = xspec.XSapec("hot"), xspec.XSvapec("cool")
        hot.kT = 1.3
        cool.kT = hot.kT
        cool.O = .2
        expression = xspec.XSphabs("absorption") * (hot + cool)
        with patch.object(type(hot), "calc", side_effect=AssertionError("model evaluated")):
            states = read_plasma_model(expression)
        self.assertEqual([s.kT for s in states], [1.3, 1.3])
        self.assertEqual(states[1].abundances["O"], .2)
        self.assertEqual(states[0].abundance_table, xspec.get_xsabund())
        self.assertIsNotNone(cool.kT.link)


if __name__ == "__main__":
    unittest.main()

