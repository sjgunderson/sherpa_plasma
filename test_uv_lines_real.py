"""Verify native-grid extraction against an independently supplied Na X table."""

import csv
import io
import os
from zipfile import ZipFile

import numpy as np
import pytest

pytest.importorskip("astropy.io.fits")

from sherpa_plasma._uv_lines import _read_native_lines
from sherpa_plasma.plasma import LineID


LINE_FILE = os.environ.get("SHERPA_PLASMA_UV_LINEFILE")
REFERENCE_ZIP = os.environ.get("SHERPA_PLASMA_UV_NA_REFERENCE")


@pytest.mark.skipif(
    not LINE_FILE or not REFERENCE_ZIP,
    reason="requires explicit APEC line FITS and sodium reference archive",
)
def test_sodium_native_grid_matches_reference_archive():
    with ZipFile(REFERENCE_ZIP) as archive:
        with archive.open("Na_X_UV_native_temperature_grid.csv") as data:
            rows = list(csv.DictReader(io.TextIOWrapper(data, encoding="utf-8")))
    expected_temperature = np.array([float(row["kT_keV"]) for row in rows])
    columns = (
        "f_emissivity_ph_cm3_s-1",
        "i_J1_emissivity_ph_cm3_s-1",
        "i_J2_emissivity_ph_cm3_s-1",
    )
    expected = np.array([[float(row[column]) for column in columns] for row in rows])

    grid = _read_native_lines(
        LINE_FILE, tuple(LineID(11, 10, upper, 1) for upper in (2, 4, 5))
    )
    complete = np.isfinite(grid.emissivities).all(axis=1)
    np.testing.assert_allclose(grid.temperatures[complete], expected_temperature,
                               rtol=1.e-14, atol=0)
    np.testing.assert_allclose(grid.emissivities[complete], expected,
                               rtol=1.e-12, atol=0)
