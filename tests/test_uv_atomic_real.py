"""Check radiative calibration against the independently supplied sodium result."""

import os

import pytest

pytest.importorskip("astropy.io.fits")

from sherpa_plasma._uv_atomic import _read_radiative_calibration


LEVEL = os.environ.get("SHERPA_PLASMA_UV_NA_LEVEL")
RADIATIVE = os.environ.get("SHERPA_PLASMA_UV_NA_RADIATIVE")


@pytest.mark.skipif(not LEVEL or not RADIATIVE, reason="requires Na X LV/LA files")
def test_sodium_atomic_calibration_matches_reference():
    result = _read_radiative_calibration(LEVEL, RADIATIVE)
    assert (result.ground, result.metastable, result.triplet_p) == (1, 2, (3, 4, 5))
    assert result.forbidden_rate == 26600
    assert result.branching == pytest.approx((0, .9891836925921211,
                                              .037142295657131834), rel=1.e-7)
    assert result.effective_branching == pytest.approx(.3503625062291136, rel=1.e-8)
    assert sum(result.redistribution) == pytest.approx(1)
    assert max(result.omitted_decay_fraction) < 1.e-12
