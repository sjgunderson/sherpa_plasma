"""Detailed checks on an explicitly supplied local database; never downloads."""
import importlib.util
import os
import unittest
import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from sherpa_plasma import AtomDB, LineID

ROOT = os.environ.get("SHERPA_PLASMA_ATOMDB")
AVAILABLE = bool(ROOT and importlib.util.find_spec("pyatomdb"))
METRICS = []


def compare(name, actual, expected, *, rtol=2.e-10, atol=1.e-29, **metadata):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    denominator = float(np.sum(np.abs(b)))
    METRICS.append({"name": name, **metadata, "values": int(a.size),
                    "maximum_absolute_difference": float(np.max(np.abs(a-b), initial=0)),
                    "relative_L1_difference": float(np.sum(np.abs(a-b))/denominator) if denominator else 0.,
                    "rtol": rtol, "atol": atol})
    assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)


@unittest.skipUnless(AVAILABLE, "requires PyAtomDB and SHERPA_PLASMA_ATOMDB local data")
class RealAtomDBValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = AtomDB(ROOT,
            linefile=os.environ.get("SHERPA_PLASMA_ATOMDB_LINEFILE", "apec_line.fits"),
            cocofile=os.environ.get("SHERPA_PLASMA_ATOMDB_COCOFILE", "apec_coco.fits"))
        with cls.db._using(cache=True) as api:
            cls.native = api.spectrum.CIESession(linefile=str(cls.db.linefile),
                cocofile=str(cls.db.cocofile), abundset=False)
            cls.native.set_eebrems(False)

    @classmethod
    def tearDownClass(cls):
        cls.native.linedata.close()
        cls.native.cocodata.close()
        cls.db.close()


    def oxygen(self):
        rows = self.db.find_lines(21.5, 22.2, element="O", ion=7, kT=.2, brightest=3)
        return tuple(LineID.from_row(row) for row in rows)

    def test_atomic_transition_and_level_metadata_across_ions(self):
        for z, stage, lo, hi in ((8, 7, 21.5, 22.2), (8, 8, 18.8, 19.1),
                                (10, 9, 13.3, 13.8), (26, 17, 14.9, 17.2), (26, 25, 1.8, 1.9)):
            with self.subTest(element=z, ion=stage):
                rows = self.db.find_lines(lo, hi, element=z, ion=stage)
                self.assertGreater(len(rows), 0)
                self.assertTrue(np.isnan(rows["Epsilon"]).all())
                for row in sorted(rows, key=lambda row: -row["Einstein_A"])[:3]:
                    info = self.db.line_info(row)
                    self.assertGreater(len(info["transitions"]), 0)
                METRICS.append({"name": "atomic_metadata", "element": z, "ion": stage,
                                "transitions_in_range": len(rows)})

    def test_line_emissivities_against_independent_native_session(self):
        for z, stage, lo, hi, peak in ((8, 7, 21.5, 22.2, .2), (8, 8, 18.8, 19.1, .5),
            (10, 9, 13.3, 13.8, .8), (26, 17, 14.9, 17.2, .6), (26, 25, 1.8, 1.9, 5.)):
            rows = self.db.find_lines(lo, hi, element=z, ion=stage, kT=peak, brightest=1)
            self.assertGreater(len(rows), 0)
            line = LineID.from_row(rows[0])
            temperatures = np.array([.7, 1., 1.3])*peak
            for abundance in (.3, 1., 2.):
                actual = self.db.line_emissivity(line, temperatures, abundances={z: abundance})
                with self.db._using():
                    self.native.set_abund(self.native.elements, 1.)
                    self.native.set_abund(z, abundance)
                    expected = self.native.return_line_emissivity(temperatures, z, stage,
                        line.upper, line.lower, apply_aeff=False, apply_abund=True)["epsilon"]
                compare("line_emissivity", actual, expected, rtol=3.e-5, atol=1.e-30,
                        element=z, ion=stage, abundance=abundance)


    def test_line_blends_and_ratios(self):
        lines = self.oxygen()
        self.assertEqual(len(set(lines)), 3)
        temperature = np.array([.15, .2, .25])
        singles = [self.db.line_emissivity(line, temperature) for line in lines]
        compare("line_blend", self.db.line_emissivity(lines, temperature), sum(singles))
        compare("line_ratio", self.db.ratio_emissivity(lines[0], lines[1], temperature),
                singles[0]/singles[1], rtol=2.e-12)
        missing = LineID(8, 7, 2000000, 1)
        self.assertTrue(np.isnan(self.db.ratio_emissivity(lines[0], missing, .2)))


    def test_general_abundances_against_independent_native_session(self):
        try:
            from sherpa.astro import xspec
        except ImportError:
            self.skipTest("requires Sherpa/XSPEC for named abundance values")
        from sherpa_plasma.plasma import _SYMBOLS

        temperature = float(self.db.temperatures[np.argmin(abs(self.db.temperatures - .2))])
        with self.db._using() as api:
            self.db._get_data(False, False, "ABUND", datacache=self.native.datacache)
            reference = api.atomdb.get_abundance(
                abundset=self.native.default_abundset, datacache=self.native.datacache,
                settings={"filemap": str(self.db.filemap), "FileMap": str(self.db.filemap),
                          "atomdbroot": str(self.db.root)})
        cases = [(name, xspec.get_xsabundances(name)) for name in
                 ("angr", "aspl", "aneb", "felc", "feld", "grsa", "lodd", "lpgp", "lpgs", "wilm")]
        custom = xspec.get_xsabundances("angr")
        custom["O"] *= .7
        before = xspec.get_xsstate()
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "composition.dat"
            np.savetxt(filename, [custom[symbol] for symbol in _SYMBOLS[1:]])
            cases += [(custom, custom),
                      ([custom[symbol] for symbol in _SYMBOLS[1:]], custom),
                      (filename, custom)]
            for spec, composition in cases:
                with self.subTest(abundset=str(spec)):
                    actual = self.db.find_lines(21.5, 22.2, kT=temperature,
                                                element="O", ion=7, abundset=spec,
                                                abundances={"O": .4})
                    with self.db._using():
                        self.native.set_abund(self.native.elements, 1.)
                        self.native.set_abund(8, .4 * composition["O"] / reference[8])
                        expected = self.native.return_linelist(
                            temperature, [21.5, 22.2], specunit="A", teunit="keV",
                            apply_aeff=False, nearest=True)
                    expected = expected[(expected["Element"] == 8) & (expected["Ion"] == 7)]
                    expected = expected[np.argsort(expected["Lambda"], kind="stable")]
                    assert_allclose(actual["Epsilon"], expected["Epsilon"], rtol=1.e-12, atol=0.)
            self.assertEqual(xspec.get_xsstate(), before)


if __name__ == "__main__":
    unittest.main()

