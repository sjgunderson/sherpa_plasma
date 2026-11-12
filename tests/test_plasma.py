"""Numerical and isolation tests using a controlled PyAtomDB test double.

These do not validate AtomDB's physical data. Real-backend tests are in
test_plasma_integration.py and require an installed local database.
"""
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sherpa_plasma import (
    AtomDB, CapturedPlasmaComponent, LineID, PlasmaLineContributions,
)
from sherpa_plasma.plasma import _LINE_DTYPE
from sherpa_plasma.units import HC_KEV_ANGSTROM as HC


def make_lines():
    rows = np.zeros(5, dtype=_LINE_DTYPE)
    rows["Element"] = [8, 8, 8, 26, 10]
    rows["Ion"] = [7, 7, 8, 17, 10]
    rows["UpperLev"] = [7, 2, 4, 3, 3]
    rows["LowerLev"] = 1
    rows["Lambda"] = [21.6, 22.1, 18.97, 15.01, 12.13]
    rows["Energy"] = HC / rows["Lambda"]
    rows["Epsilon"] = [4.e-15, 2.e-15, 1.e-15, 3.e-15, 2.e-15]
    rows["Einstein_A"] = np.nan
    return rows


class FakeHDU(list):
    def __init__(self, data=None, header=None):
        super().__init__([SimpleNamespace(header=header or {}), SimpleNamespace(data=data)])
        self.closed = False

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, *, linefile, cocofile, abundset):
        assert abundset is False
        assert Path(os.environ["ATOMDB"], "userdata").exists()
        assert Path(linefile).is_file() and Path(cocofile).is_file()
        self.spectra = SimpleNamespace(kTlist=np.array([.1, 1., 10.]), datacache={})
        self.elements = list(range(1, 31))
        self.abund = np.ones(31)
        self.abundsetvector = np.ones(31)
        self.abundset = self.default_abundset = "AG89"
        self.datacache = {}
        self.linedata = FakeHDU(header={"CHECKSUM": "LINECHECK"})
        self.cocodata = FakeHDU(header={"CHECKSUM": "COCOCHECK"})
        self.line_calls = 0
        self.nearest_requests = []

    def set_abund(self, elements, value):
        self.abund[elements] = value

    def set_eebrems(self, flag):
        self.do_eebrems = flag

    def set_abundset(self, name):
        assert "ABUND" in self.datacache["data"]["misc"]
        self.abundset = name
        self.abundsetvector[:] = 1. if name == "AG89" else 2.


    def return_linelist(self, kT, specrange, *, specunit, teunit, apply_aeff, nearest):
        self.nearest_requests.append(("linelist", kT, nearest))
        assert teunit == "keV" and not apply_aeff
        self.line_calls += 1
        if kT == .1 and not nearest:
            raise IndexError("simulated PyAtomDB lowest-temperature edge case")
        rows = make_lines()
        rows["Epsilon"] *= kT * self.abund[rows["Element"]] * self.abundsetvector[rows["Element"]]
        x = rows["Lambda"] if specunit == "A" else rows["Energy"]
        return rows[(x >= specrange[0]) & (x <= specrange[1])]


class FakeNEISession(FakeSession):
    def _lines(self, kT, age_factor, specrange, specunit):
        rows = make_lines()
        rows["Epsilon"] *= (
            kT * age_factor * self.abund[rows["Element"]]
            * self.abundsetvector[rows["Element"]]
        )
        x = rows["Lambda"] if specunit in ("A", "Angstrom") else rows["Energy"]
        return rows[(x >= specrange[0]) & (x <= specrange[1])]

    def return_linelist(
        self, kT, tau, specrange, *, specunit="A", tau_l=0., **kwargs
    ):
        factor = (tau - tau_l) / 1.e11 if tau_l else tau / 1.e11
        return self._lines(kT, factor, specrange, specunit)

    def return_line_emissivity(
        self, kT, tau, z, ion, upper, lower, *, tau_llist=0.,
        init_pop="ionizing", **kwargs,
    ):
        row = self._lines(
            kT, (tau - tau_llist) / 1.e11 if tau_llist else tau / 1.e11,
            [0., 1.e10], "A",
        )
        match = row[
            (row["Element"] == z) & (row["Ion"] == ion)
            & (row["UpperLev"] == upper) & (row["LowerLev"] == lower)
        ]
        return {"epsilon": float(match["Epsilon"].sum())}


class PlasmaTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temp).resolve()
        for name in ("filemap", "apec_line.fits", "apec_coco.fits",
                     "apec_nei_line.fits", "apec_nei_comp.fits"):
            (self.root / name).touch()
        (self.root / "VERSION").write_text("test-version")
        self.calls = []
        self.api = SimpleNamespace(
            atomdb=SimpleNamespace(get_data=self.get_data, read_filemap=self.read_filemap,
                                   get_abundance=self.get_abundance),
            spectrum=SimpleNamespace(
                CIESession=FakeSession,
                NEISession=FakeNEISession,
                PShockSession=FakeNEISession,
            ),
            atomic=SimpleNamespace(Z_to_mass=lambda *a, **k: np.arange(31) * 2.))
        self.stack.enter_context(patch.dict(sys.modules, {"pyatomdb": self.api}))
        self.db = self.stack.enter_context(AtomDB(self.root))

    def get_abundance(self, *, abundset, datacache, settings):
        self.assertEqual(abundset, "AG89")
        self.assertIn("ABUND", datacache["data"]["misc"])
        return {z: 1. for z in range(1, 31)}

    def read_filemap(self, filename, *, atomdbroot):
        self.assertEqual(filename, str(self.root / "filemap"))
        self.assertEqual(atomdbroot, str(self.root))
        return {"Z": [8, 8, 26, 10], "z1": [7, 8, 17, 10],
                "la": ["o7.fits", "o8.fits", "fe17.fits", "ne10.fits"]}

    def get_data(self, z, ion, kind, *, offline, datacache, settings):
        self.assertTrue(offline)
        self.assertEqual(os.environ["ATOMDB"], str(self.root))
        self.assertEqual(settings, {"filemap": str(self.root / "filemap"),
                                   "FileMap": str(self.root / "filemap"), "atomdbroot": str(self.root)})
        self.calls.append((z, ion, kind))
        if kind == "ABUND":
            result = FakeHDU(np.array(
                [("AG89",), ("other",), ("Wilm00",)],
                dtype=[("Source", "U20")],
            ))
            datacache.setdefault("data", {}).setdefault("misc", {})["ABUND"] = result
            return result
        if kind == "EIGEN":
            result = FakeHDU()
            datacache.setdefault("data", {}).setdefault(
                "misc", {}
            ).setdefault("EIGEN", {})[z] = result
            return result
        if kind == "LV":
            return FakeHDU(np.array([(i,) for i in range(1, 10)], dtype=[("level", "i4")]))
        rows = make_lines()
        rows = rows[(rows["Element"] == z) & (rows["Ion"] == ion)]
        result = np.zeros(len(rows), dtype=[("upper_lev", "i4"), ("lower_lev", "i4"),
                                           ("wavelen", "f8"), ("wave_obs", "f8"), ("einstein_a", "f8")])
        for src, dst in [("UpperLev", "upper_lev"), ("LowerLev", "lower_lev"), ("Lambda", "wavelen")]:
            result[dst] = rows[src]
        result["einstein_a"] = 1.e10
        return FakeHDU(result)

    def test_import_and_connection_are_lazy(self):
        self.assertIsNone(self.db._api)
        self.assertIsNone(self.db._scratch)
        self.assertIsNone(self.db._session)


    def test_atomic_search_needs_no_emissivity_files(self):
        (self.root / "apec_line.fits").unlink()
        (self.root / "apec_coco.fits").unlink()
        rows = self.db.find_lines(21.5, 22.2, element="O", ion=7)
        assert_array_equal(rows["UpperLev"], [7, 2])
        self.assertTrue(np.isnan(rows["Epsilon"]).all())
        self.assertTrue(np.isnan(rows["ModelContribution"]).all())
        assert_array_equal(rows["Einstein_A"], [1.e10, 1.e10])
        self.assertIsNone(self.db._session)

    def test_search_energy_filter_ranking_and_units(self):
        rows = self.db.find_lines(.5, .7, unit="keV", element=8, ion=7,
                                  kT=1., brightest=1, min_emissivity=1.e-15)
        self.assertEqual(LineID.from_row(rows[0]), LineID(8, 7, 7, 1))
        self.assertTrue(np.isnan(rows["Einstein_A"]).all())
        self.assertTrue(np.isnan(rows["ModelContribution"]).all())
        assert_allclose(rows["Energy"], HC / rows["Lambda"])
        nm = self.db.find_lines(2.15, 2.22, unit="nm", element="O", kT=1.)
        self.assertEqual(len(nm), 2)

    def test_line_info_preserves_one_based_levels(self):
        info = self.db.line_info(LineID("O", 7, 7, 1))
        self.assertEqual(info["upper"]["level"], 7)
        self.assertEqual(info["lower"]["level"], 1)
        info["transitions"]["wavelen"] = 0
        self.assertGreater(self.db.transitions("O", 7)["wavelen"][0], 0)

    def test_tabulated_temperatures_use_tables_without_interpolating_neighboring_lines(self):
        self.db.find_lines(.5, .7, unit="keV", kT=1.)
        self.assertTrue(all(nearest for _, _, nearest in self.db._cie().nearest_requests))
        self.db._cie().nearest_requests.clear()
        self.db.find_lines(.5, .7, unit="keV", kT=1.+1.e-7)
        self.assertTrue(all(not nearest for _, _, nearest in self.db._cie().nearest_requests))

    @unittest.skipUnless(importlib.util.find_spec("astropy") is not None, "requires Astropy FITS records")
    def test_line_info_copies_astropy_fits_rows_without_mutating_the_table(self):
        from astropy.io import fits
        levels = fits.BinTableHDU.from_columns([
            fits.Column(name="level", format="J", array=np.arange(1, 10))]).data
        with patch.object(AtomDB, "levels", return_value=levels):
            info = self.db.line_info(LineID("O", 7, 7, 1))
        self.assertEqual(info["upper"]["level"], 7)
        self.assertEqual(info["lower"]["level"], 1)
        info["upper"]["level"] = 99
        info["lower"]["level"] = 88
        assert_array_equal(levels["level"], np.arange(1, 10))

    def test_atomic_wavelength_prefers_positive_observed_value(self):
        original = self.get_data
        def observed_data(*args, **kwargs):
            data = original(*args, **kwargs)
            if args[2] == "LA" and args[:2] == (8, 7):
                data[1].data["wavelen"][0] = 100.
                data[1].data["wave_obs"] = [21.6, np.nan]
            return data
        with patch.object(self.api.atomdb, "get_data", side_effect=observed_data):
            rows = self.db.find_lines(21.5, 22.2, element=8, ion=7)
        assert_allclose(rows["Lambda"], [21.6, 22.1])

    def test_emissivity_scalar_blend_vector_and_kelvin(self):
        r, f = LineID(8, 7, 7, 1), LineID(8, 7, 2, 1)
        self.assertAlmostEqual(self.db.line_emissivity(r, 1.), 4.e-15)
        assert_allclose(self.db.line_emissivity([r, f], [.1, 1., 2.]), [6.e-16, 6.e-15, 12.e-15])
        kelvin = 1.602176634e-16 / 1.380649e-23
        assert_allclose(self.db.line_emissivity(r, kelvin, teunit="K"), 4.e-15)
        assert_allclose(self.db.ratio_emissivity(r, f, [1., 2.]), 2.)
        self.assertTrue(np.isnan(self.db.ratio_emissivity(r, LineID(8, 7, 99, 1), 1.)))
        with self.assertRaises(ValueError):
            self.db.line_emissivity([r, r], 1.)

    def test_explicit_nei_and_pshock_line_apis(self):
        line = LineID(8, 7, 7, 1)
        assert_allclose(
            self.db.nei_line_emissivity(
                line, 1., 1.e11, abundances={"O": .5}
            ),
            2.e-15,
        )
        assert_allclose(
            self.db.nei_line_emissivity(line, [1., 2.], 1.e11),
            [4.e-15, 8.e-15],
        )
        assert_allclose(
            self.db.pshock_line_emissivity(
                line, 1., 3.e11, tau_lower=1.e11
            ),
            8.e-15,
        )

        nei_rows = self.db.find_nei_lines(
            21.5, 22.2, temperature=1., ionization_age=1.e11,
            element="O", ion=7, brightest=1,
        )
        self.assertEqual(LineID.from_row(nei_rows[0]), line)
        assert_allclose(nei_rows["Epsilon"], [4.e-15])
        shock_rows = self.db.find_pshock_lines(
            21.5, 22.2, temperature=1., tau_lower=1.e11,
            tau_upper=3.e11, element="O", ion=7, brightest=1,
        )
        assert_allclose(shock_rows["Epsilon"], [8.e-15])

        with self.assertRaisesRegex(ValueError, "ionization age"):
            self.db.nei_line_emissivity(line, 1., -1.)
        with self.assertRaisesRegex(ValueError, "NEI emissivity range"):
            self.db.nei_line_emissivity(line, .01, 1.e11)
        with self.assertRaisesRegex(ValueError, "must exceed"):
            self.db.pshock_line_emissivity(
                line, 1., 1.e11, tau_lower=1.e11
            )

    def test_nei_loads_local_eigen_data_without_userdata(self):
        self.assertFalse((self.root / "userdata").exists())
        initial_files = {path.name for path in self.root.iterdir()}

        with patch.object(self.db, "_find_sherpa_eigen_file", return_value=None):
            rows = self.db.find_nei_lines(
                21.5, 22.2, temperature=1., ionization_age=1.e11,
                element="O", ion=7, brightest=1,
            )

        self.assertGreater(len(rows), 0)
        eigen_requests = [call for call in self.calls if call[2] == "EIGEN"]
        self.assertEqual(len(eigen_requests), len(self.db._nei().elements))
        self.assertEqual(
            set(self.db._nei().datacache["data"]["misc"]["EIGEN"]),
            set(self.db._nei().elements),
        )
        self.assertFalse((self.root / "userdata").exists())
        self.assertEqual({path.name for path in self.root.iterdir()}, initial_files)

    def test_sherpa_combined_eigen_file_preloads_every_element(self):
        eigen_path = self.root / "eigen_v3.0.4.fits"
        eigen_path.touch()

        class FakeCombinedHDUList(list):
            def __init__(self, names):
                super().__init__(
                    [SimpleNamespace(name=name, header={"DATASUM": "sum"})
                     for name in names]
                )
                self._file = SimpleNamespace(closed=False)

            def close(self):
                self._file.closed = True

        symbols = ("H", "He", "Li", "Be", "B", "C", "N", "O",
                   "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
                   "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
                   "Mn", "Fe", "Co", "Ni", "Cu", "Zn")
        opened = FakeCombinedHDUList(
            ["PRIMARY", *(symbol.lower() for symbol in symbols)]
        )
        fake_fits = ModuleType("astropy.io.fits")
        fake_fits.open = lambda path, memmap: opened
        astropy = ModuleType("astropy")
        astropy_io = ModuleType("astropy.io")
        astropy.io = astropy_io
        astropy_io.fits = fake_fits

        with patch.dict(sys.modules, {
            "astropy": astropy, "astropy.io": astropy_io,
            "astropy.io.fits": fake_fits,
        }), patch.object(
            self.db, "_find_sherpa_eigen_file", return_value=eigen_path
        ):
            session = self.db._nei()

        eigen_cache = session.datacache["data"]["misc"]["EIGEN"]
        self.assertEqual(set(eigen_cache), set(session.elements))
        self.assertTrue(all(value is self.db._sherpa_eigen_hdul
                            for value in eigen_cache.values()))
        self.assertEqual(self.db._sherpa_eigen_path, eigen_path)
        self.assertFalse(any(call[2] == "EIGEN" for call in self.calls))
        self.assertFalse((self.root / "userdata").exists())

        opened = self.db._sherpa_eigen_hdul
        self.db.close()
        self.assertTrue(opened._file.closed)
        self.assertIsNone(self.db._sherpa_eigen_hdul)

    def test_sherpa_eigen_discovery_prefers_latest_in_headas_tree(self):
        install = self.root / "heasoft"
        headas = install / "aarch64-apple-darwin"
        model_data = install / "spectral" / "modelData"
        model_data.mkdir(parents=True)
        older = model_data / "eigen_v3.0.fits"
        newest = model_data / "eigen_v3.0.4.fits"
        older.touch()
        newest.touch()

        with patch.dict(os.environ, {"HEADAS": str(headas)}), \
             patch.object(sys, "prefix", str(self.root / "isolated-prefix")):
            found = self.db._find_sherpa_eigen_file()

        self.assertEqual(found, newest.resolve())

    def test_missing_local_eigen_data_fails_without_userdata_fallback(self):
        original = self.api.atomdb.get_data

        def without_eigen(z, ion, kind, **kwargs):
            if kind == "EIGEN":
                raise KeyError(z)
            return original(z, ion, kind, **kwargs)

        with patch.object(self.db, "_find_sherpa_eigen_file", return_value=None), \
             patch.object(self.api.atomdb, "get_data", side_effect=without_eigen):
            with self.assertRaisesRegex(
                FileNotFoundError, "no EIGEN data for H \\(Z=1\\)"
            ):
                self.db.nei_line_emissivity(LineID(8, 7, 7, 1), 1., 1.e11)

        self.assertFalse((self.root / "userdata").exists())
        self.assertEqual(self.db._nei_sessions, {})

    def test_lowest_temperature_endpoint_and_range_rejection(self):
        self.assertEqual(len(self.db.find_lines(21., 23., kT=.1)), 2)
        self.db.find_lines(.5, .7, unit="keV", kT=.1)
        for t in [.09, 10.1, np.inf, -1.]:
            with self.assertRaises(ValueError):
                self.db.find_lines(.5, .7, unit="keV", kT=t)

    def test_abundances_do_not_leak_between_calls(self):
        line = LineID(8, 7, 7, 1)
        assert_allclose(self.db.line_emissivity(line, 1., abundances={"O": .5}), 2.e-15)
        assert_allclose(self.db.line_emissivity(line, 1.), 4.e-15)
        with self.assertRaises(ValueError):
            self.db.line_emissivity(line, 1., abundances={"O": 1., 8: 2.})
        self.db.abundset = "other"
        self.db._session = None
        assert_allclose(self.db.line_emissivity(line, 1.), 8.e-15)

    @staticmethod
    def captured_component(name, parameters, temperatures=None, weights=None, *,
                           model_type="xsmkcflow", abundance_table="angr"):
        return CapturedPlasmaComponent(
            name=name,
            model_type=model_type,
            parameters=parameters,
            abundance_table=abundance_table,
            xset={},
            temperatures=temperatures,
            weights=weights,
            abundance_values={z: 2. if abundance_table == "wilm" else 1.
                              for z in range(1, 31)},
        )

    def test_captured_nei_and_pshock_components_are_ranked(self):
        nei = self.captured_component(
            "nei", {"kT": 1., "Tau": 1.e11}, model_type="xsnei"
        )
        shock = self.captured_component(
            "shock",
            {"kT": 1., "Tau_l": 1.e11, "Tau_u": 3.e11},
            model_type="xspshock",
        )
        result = self.db.query_plasma_lines(
            21.5, 22.2, components=(nei, shock), brightest=2
        )
        self.assertEqual(len(result.components), 2)
        self.assertEqual(
            [item.model_type for item in result.components],
            ["xsnei", "xspshock"],
        )
        assert_allclose(result.components[0].values, [4.e-15, 2.e-15])
        assert_allclose(result.components[1].values, [8.e-15, 4.e-15])
        assert_allclose(result.total, [12.e-15, 6.e-15])
        assert_allclose(result.lines["ModelContribution"], result.total)
        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model",
            return_value=(nei, shock),
        ):
            ranked = self.db.find_lines(21.5, 22.2, brightest=1)
        assert_allclose(ranked["ModelContribution"], [12.e-15])

    def test_captured_query_weights_components_and_ranks_total(self):
        first = self.captured_component(
            "flow", {"Abundanc": .5}, [.1, 1.], [2., 3.]
        )
        second = self.captured_component(
            "hot", {"Abundanc": 1.}, [1.], [1.]
        )
        temperatures_before = first.temperatures.copy()
        weights_before = first.weights.copy()

        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model"
        ) as capture:
            result = self.db.query_plasma_lines(
                21.5, 22.2, components=[first, second]
            )
        capture.assert_not_called()

        self.assertIsInstance(result, PlasmaLineContributions)
        assert_array_equal(
            [LineID.from_row(row) for row in result.lines],
            [LineID(8, 7, 7, 1), LineID(8, 7, 2, 1)],
        )
        assert_allclose(result.components[0].values, [6.4e-15, 3.2e-15])
        assert_allclose(result.components[1].values, [4.e-15, 2.e-15])
        assert_allclose(result.total, [10.4e-15, 5.2e-15])
        assert_allclose(result.lines["ModelContribution"], result.total)
        assert_array_equal(result.lines["Epsilon"], [np.nan, np.nan])
        self.assertFalse(result.total.flags.writeable)
        assert_array_equal(first.temperatures, temperatures_before)
        assert_array_equal(first.weights, weights_before)

        brightest = self.db.query_plasma_lines(
            21.5, 22.2, components=[first, second], brightest=1
        )
        self.assertEqual(len(brightest.lines), 1)
        self.assertEqual(LineID.from_row(brightest.lines[0]), LineID(8, 7, 7, 1))
        assert_allclose(brightest.total, [10.4e-15])

        with patch.object(
            self.db, "_line_emissivity_by_line",
            return_value=np.full((2, 2), 1.e-15),
        ):
            tied = self.db.query_plasma_lines(
                21.5, 22.2, components=[first], brightest=1
            )
        self.assertEqual(LineID.from_row(tied.lines[0]), LineID(8, 7, 7, 1))

    def test_find_lines_ranks_active_model_total_in_model_contribution(self):
        first = self.captured_component(
            "flow", {"Abundanc": .5}, [.1, 1.], [2., 3.]
        )
        second = self.captured_component(
            "hot", {"Abundanc": 1.}, [1.], [1.]
        )
        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model",
            return_value=(first, second),
        ) as capture:
            rows = self.db.find_lines(
                21.5, 22.2, element="O", ion=7, brightest=1
            )
        capture.assert_called_once_with(None, dataset_id=None, session=None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(LineID.from_row(rows[0]), LineID(8, 7, 7, 1))
        self.assertTrue(np.isnan(rows["Epsilon"]).all())
        assert_allclose(rows["ModelContribution"], [10.4e-15])

    def test_find_lines_model_source_selectors_and_threshold(self):
        component = self.captured_component(
            "hot", {"Abundanc": 1.}, [1.], [1.]
        )
        model = SimpleNamespace()
        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model",
            return_value=(component,),
        ) as capture:
            rows = self.db.find_lines(
                21.5, 22.2, unit="Angstrom", brightest=5,
                min_emissivity=3.e-15, model=model,
            )
        capture.assert_called_once_with(model, dataset_id=None, session=None)
        assert_allclose(rows["ModelContribution"], [4.e-15])

        session = SimpleNamespace()
        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model",
            return_value=(component,),
        ) as capture:
            self.db.find_lines(
                21.5, 22.2, brightest=5, dataset_id="grating",
                session=session,
            )
        capture.assert_called_once_with(
            None, dataset_id="grating", session=session
        )

    def test_find_lines_rejects_model_only_options_without_model_ranking(self):
        with self.assertRaisesRegex(ValueError, "only valid when ranking"):
            self.db.find_lines(21.5, 22.2, model=SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "only valid when ranking"):
            self.db.find_lines(
                21.5, 22.2, model=SimpleNamespace(), kT=1., brightest=1
            )
        with self.assertRaisesRegex(ValueError, "taken from the captured model"):
            self.db.find_lines(
                21.5, 22.2, abundances={"O": 1.}, brightest=1
            )

    def test_captured_query_discovers_union_over_temperature_grid(self):
        component = self.captured_component(
            "flow", {"Abundanc": 1.}, [.1, 1.], [2., 3.]
        )
        original = self.db._cie().return_linelist

        def temperature_lines(kT, *args, **kwargs):
            rows = original(kT, *args, **kwargs)
            upper = 7 if kT == .1 else 2
            return rows[rows["UpperLev"] == upper]

        with patch.object(self.db._cie(), "return_linelist",
                          side_effect=temperature_lines):
            result = self.db.query_plasma_lines(
                21.5, 22.2, components=[component]
            )

        assert_array_equal(
            [LineID.from_row(row) for row in result.lines],
            [LineID(8, 7, 7, 1), LineID(8, 7, 2, 1)],
        )
        assert_allclose(result.components[0].values, [.8e-15, 6.e-15])

    def test_captured_elemental_abundance_suppresses_zero_and_scales_lines(self):
        absent = self.captured_component(
            "variable", {"Ne": 0.}, [1.], [1.], model_type="xsvapec"
        )
        result = self.db.query_plasma_lines(12., 12.3, components=[absent])
        self.assertEqual(len(result.lines), 0)
        self.assertEqual(len(result.components[0].values), 0)

        reduced = self.captured_component(
            "variable", {"Ne": .25}, [1.], [1.], model_type="xsvapec"
        )
        result = self.db.query_plasma_lines(12., 12.3, components=[reduced])
        assert_array_equal(
            [LineID.from_row(row) for row in result.lines],
            [LineID(10, 10, 3, 1)],
        )
        assert_allclose(result.total, [.5e-15])

        explicit_override = self.captured_component(
            "flow", {"Abundanc": 0., "Ne": 1.}, [1.], [1.]
        )
        result = self.db.query_plasma_lines(
            12., 12.3, components=[explicit_override]
        )
        self.assertEqual(len(result.lines), 1)
        assert_allclose(result.total, [2.e-15])

    def test_captured_abundance_baseline_maps_and_does_not_leak(self):
        component = self.captured_component(
            "hot", {"Abundanc": 1.}, [1.], [1.], abundance_table="wilm"
        )
        result = self.db.query_plasma_lines(
            21.5, 22.2, components=[component]
        )
        assert_allclose(result.total, [8.e-15, 4.e-15])
        assert_allclose(self.db._session.abundsetvector[1:], 2.)

        line = LineID(8, 7, 7, 1)
        self.assertEqual(self.db.line_emissivity(line, 1.), 4.e-15)
        self.assertEqual(self.db._session.abundset, "AG89")

    def test_captured_query_accepts_numeric_baselines_and_checks_limits(self):
        for table in ("aspl", "lpgp", "lodd", "file", "another_table"):
            result = self.db.query_plasma_lines(
                21.5, 22.2, components=[self.captured_component(
                    "hot", {"Abundanc": 1.}, [1.], [1.],
                    abundance_table=table,
                )],
            )
            assert_allclose(result.total, [4.e-15, 2.e-15])
        component = self.captured_component(
            "hot", {"Abundanc": 1.}, [1.], [1.]
        )
        for value in (0, True, 1.5):
            with self.assertRaises(ValueError):
                self.db.query_plasma_lines(
                    21.5, 22.2, components=[component], brightest=value
                )
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.db.query_plasma_lines(21.5, 22.2, components=[])

    def test_query_captures_default_model_or_explicit_source(self):
        component = self.captured_component(
            "hot", {"Abundanc": 1.}, [1.], [1.]
        )
        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model",
            return_value=(component,),
        ) as capture:
            result = self.db.query_plasma_lines(21.5, 22.2)
            self.assertEqual(len(result.lines), 2)
            capture.assert_called_once_with(
                None, dataset_id=None, session=None
            )

        model = SimpleNamespace()
        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model",
            return_value=(component,),
        ) as capture:
            self.db.query_plasma_lines(21.5, 22.2, model=model)
            capture.assert_called_once_with(
                model, dataset_id=None, session=None
            )

        session = SimpleNamespace()
        with patch(
            "sherpa_plasma.xspec_capture.capture_plasma_model",
            return_value=(component,),
        ) as capture:
            self.db.query_plasma_lines(
                21.5, 22.2, dataset_id="grating", session=session
            )
            capture.assert_called_once_with(
                None, dataset_id="grating", session=session
            )

    def test_query_rejects_conflicting_sources_and_invalid_captures(self):
        component = self.captured_component(
            "hot", {"Abundanc": 1.}, [1.], [1.]
        )
        with self.assertRaisesRegex(ValueError, "model or a dataset/session"):
            self.db.query_plasma_lines(
                21.5, 22.2, model=object(), dataset_id=1
            )
        with self.assertRaisesRegex(ValueError, "captured components or"):
            self.db.query_plasma_lines(
                21.5, 22.2, components=[component], session=object()
            )
        with self.assertRaisesRegex(TypeError, "CapturedPlasmaComponent"):
            self.db.query_plasma_lines(
                21.5, 22.2, components=[object()]
            )

    def test_environment_restored_and_database_not_modified(self):
        original = sorted(p.name for p in self.root.iterdir())
        with patch.dict(os.environ, {"ATOMDB": "prior-db", "OLDATOMDB": "prior-old"}):
            self.db.find_lines(.5, .7, unit="keV", kT=1.)
            self.assertEqual(os.environ["ATOMDB"], "prior-db")
            self.assertEqual(os.environ["OLDATOMDB"], "prior-old")
            def bad(*args, **kwargs):
                os.environ["OLDATOMDB"] = "changed-in-backend"
                raise RuntimeError("broken filemap")
            with patch.object(self.api.atomdb, "read_filemap", bad):
                with self.assertRaises(RuntimeError):
                    self.db.find_lines(10., 30.)
            self.assertEqual(os.environ["ATOMDB"], "prior-db")
            self.assertEqual(os.environ["OLDATOMDB"], "prior-old")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), original)

    def test_info_and_cleanup(self):
        self.db.find_lines(.5, .7, unit="keV", kT=1.)
        self.assertEqual(self.db.info["version"], "test-version")
        self.assertEqual(self.db.info["line_checksum"], "LINECHECK")
        s, cache = self.db._session, Path(self.db._scratch.name)
        self.db.close()
        self.assertTrue(s.linedata.closed and s.cocodata.closed)
        self.assertFalse(cache.exists())
        with self.assertRaises(RuntimeError):
            self.db.find_lines(10., 20.)

    def test_missing_local_files_fail_without_initialization(self):
        (self.root / "apec_line.fits").unlink()
        with self.assertRaises(FileNotFoundError):
            self.db.find_lines(.5, .7, unit="keV", kT=1.)
        with patch.object(self.api.atomdb, "get_data", return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.db.find_lines(10., 30.)

    def test_missing_nei_files_fail_without_initializing_cie_or_nei_sessions(self):
        (self.root / "apec_nei_line.fits").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "apec_nei_line.fits"):
            self.db.nei_line_emissivity(LineID(8, 7, 7, 1), 1., 1.e11)
        self.assertIsNone(self.db._session)
        self.assertEqual(self.db._nei_sessions, {})

    def test_path_override_ignores_another_global_root(self):
        with patch.dict(os.environ, {"ATOMDB": "/wrong/root"}):
            with AtomDB(self.root, linefile="$ATOMDB/apec_line.fits") as db:
                self.assertEqual(db.linefile, self.root / "apec_line.fits")

    def test_bad_selection_inputs(self):
        for lo, hi in [(0, 10), (-1, 1), (22, 21), (1, np.inf)]:
            with self.assertRaises(ValueError):
                self.db.find_lines(lo, hi)
        for element in ["", "Og", 0, 31, True, 8.1]:
            with self.assertRaises(ValueError):
                self.db.find_lines(10, 30, element=element)
        for option in [{"brightest": 0}, {"min_emissivity": 0}, {"abundances": {8: 1}}]:
            with self.assertRaises(ValueError):
                self.db.find_lines(10, 30, **option)


if __name__ == "__main__":
    unittest.main()

