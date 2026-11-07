"""Optional real PyAtomDB/APED checks. Never downloads data."""
import importlib.util
import os
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from sherpa_plasma import AtomDB, LineID

ROOT = os.environ.get("SHERPA_PLASMA_ATOMDB")
REAL_DB = bool(ROOT and importlib.util.find_spec("pyatomdb"))
NEI_DB = bool(
    REAL_DB and all(
        (Path(ROOT) / name).is_file()
        for name in ("apec_nei_line.fits", "apec_nei_comp.fits")
    )
)


@unittest.skipUnless(REAL_DB, "requires PyAtomDB and SHERPA_PLASMA_ATOMDB local data")
class AtomDBIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = AtomDB(ROOT,
            linefile=os.environ.get("SHERPA_PLASMA_ATOMDB_LINEFILE", "apec_line.fits"),
            cocofile=os.environ.get("SHERPA_PLASMA_ATOMDB_COCOFILE", "apec_coco.fits"))

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_atomic_search_and_level_metadata(self):
        rows = self.db.find_lines(21.5, 22.2, element="O", ion=7)
        self.assertGreaterEqual(len(rows), 3)
        info = self.db.line_info(LineID.from_row(rows[0]))
        self.assertGreater(len(info["transitions"]), 0)
        self.assertTrue(np.isnan(rows["Epsilon"]).all())

    def test_individual_emissivities_agree_with_native_pyatomdb(self):
        lo, hi = self.db.temperature_range
        temperature = np.sqrt(lo * hi)
        rows = self.db.find_lines(21.5, 22.2, kT=temperature, element="O", ion=7, brightest=1)
        self.assertGreater(len(rows), 0)
        line = LineID.from_row(rows[0])
        actual = self.db.line_emissivity(line, temperature)
        with self.db._using():
            result = self.db._cie().return_line_emissivity(
                temperature, line.element, line.ion, line.upper, line.lower,
                apply_aeff=False, teunit="keV")
        assert_allclose(actual, result["epsilon"], rtol=2.e-5, atol=1.e-35)

    def test_temperature_knots_and_endpoints(self):
        temperatures = self.db.temperatures
        self.assertTrue(np.all(np.diff(temperatures) > 0))
        self.assertEqual(self.db.temperature_range, (temperatures[0], temperatures[-1]))
        temperatures[:] = 0
        self.assertGreater(self.db.temperatures[0], 0)
        for temperature in self.db.temperature_range:
            self.db.find_lines(21.5, 22.2, kT=temperature, element="O")
        with self.assertRaises(ValueError):
            self.db.find_lines(21.5, 22.2, kT=self.db.temperature_range[0] / 2)

    def test_search_and_emissivities_need_no_network(self):
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network access")):
            rows = self.db.find_lines(21.5, 22.2, element="O", ion=7, kT=.2, brightest=2)
            self.assertGreater(len(rows), 0)
            self.assertGreater(self.db.line_emissivity(LineID.from_row(rows[0]), .2), 0)

    @unittest.skipUnless(NEI_DB, "requires local AtomDB NEI emissivity tables")
    def test_nei_and_pshock_emissivities_match_native_sessions(self):
        nei_rows = self.db.find_nei_lines(
            21.5, 22.2, element="O", ion=7, temperature=1.,
            ionization_age=1.e11, brightest=1,
        )
        self.assertGreater(len(nei_rows), 0)
        line = LineID.from_row(nei_rows[0])
        nei_value = self.db.nei_line_emissivity(line, 1., 1.e11)
        with self.db._using():
            native = self.db._nei().return_line_emissivity(
                1., 1.e11, line.element, line.ion, line.upper, line.lower,
                teunit="keV", apply_aeff=False,
            )["epsilon"]
        assert_allclose(nei_value, native, rtol=2.e-5, atol=1.e-35)

        pshock_value = self.db.pshock_line_emissivity(
            line, 1., 1.e11, tau_lower=1.e10
        )
        with self.db._using():
            native = self.db._nei(pshock=True).return_line_emissivity(
                1., 1.e11, line.element, line.ion, line.upper, line.lower,
                tau_llist=1.e10, teunit="keV", apply_aeff=False,
            )["epsilon"]
        assert_allclose(pshock_value, native, rtol=2.e-5, atol=1.e-35)


if __name__ == "__main__":
    unittest.main()
