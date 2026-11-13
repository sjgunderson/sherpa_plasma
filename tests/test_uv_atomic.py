"""Tests for identifying and validating He-like radiative inputs."""

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from sherpa_plasma._uv_atomic import _read_radiative_calibration


def _write_inputs(tmp_path, *, extra=0.0, cascade=0.0, metastable_extra=0.0,
                  self_decay=0.0, ground2=1.0, duplicate=False):
    configs = ["1s2", "1s1 2s1", "1s1 2s1", "1s1 2p1", "1s1 2p1", "1s1 2p1"]
    levels = fits.BinTableHDU.from_columns([
        fits.Column(name="ELEC_CONFIG", format="20A", array=configs),
        fits.Column(name="N_QUAN", format="J", array=[1, 2, 2, 2, 2, 2]),
        fits.Column(name="L_QUAN", format="J", array=[0, 0, 0, 1, 1, 1]),
        fits.Column(name="S_QUAN", format="E", array=[0, 1, 0, 1, 1, 1]),
        fits.Column(name="LEV_DEG", format="J", array=[1, 3, 1, 1, 3, 5]),
        fits.Column(name="ENERGY", format="E", array=[0, 1, 2, 3, 4, 5]),
    ])
    transitions = [(2, 1, 10), (4, 2, 100), (5, 1, 300),
                   (5, 2, 100), (6, 1, ground2), (6, 2, 99), (6, 3, extra),
                   (6, 5, cascade), (6, 6, self_decay), (2, 3, metastable_extra)]
    if duplicate:
        transitions.append((2, 1, 10))
    upper, lower, rate = zip(*transitions)
    radiative = fits.BinTableHDU.from_columns([
        fits.Column(name="UPPER_LEV", format="J", array=upper),
        fits.Column(name="LOWER_LEV", format="J", array=lower),
        fits.Column(name="EINSTEIN_A", format="E", array=rate),
    ])
    for hdu in (levels, radiative):
        hdu.header["ELEMENT"] = 8
        hdu.header["ION_STAT"] = 6
        hdu.header["ION_NAME"] = "O VII"
    level_path, radiative_path = tmp_path / "levels.fits", tmp_path / "rates.fits"
    fits.HDUList([fits.PrimaryHDU(), levels]).writeto(level_path)
    fits.HDUList([fits.PrimaryHDU(), radiative]).writeto(radiative_path)
    return level_path, radiative_path


def test_identifies_oxygen_style_indices_and_branch_weights(tmp_path):
    result = _read_radiative_calibration(*_write_inputs(tmp_path))
    assert (result.element, result.ion) == (8, 7)
    assert (result.ground, result.metastable, result.triplet_p) == (1, 2, (4, 5, 6))
    assert result.forbidden_rate == 10
    assert result.branching == pytest.approx((0, .75, .01))
    assert result.effective_branching == pytest.approx(.75 / 3 + .01 * 5 / 9)
    assert sum(result.redistribution) == pytest.approx(1)
    np.testing.assert_equal(result.omitted_decay_fraction, (0, 0, 0))


@pytest.mark.parametrize("extra", [1., 1.e-9, -1., float("nan")])
def test_rejects_significant_or_invalid_extra_channels(tmp_path, extra):
    with pytest.raises(ValueError, match="additional radiative decay"):
        _read_radiative_calibration(*_write_inputs(tmp_path, extra=extra))


def test_rejects_duplicate_transition(tmp_path):
    with pytest.raises(ValueError, match="radiative transition"):
        _read_radiative_calibration(*_write_inputs(tmp_path, duplicate=True))


@pytest.mark.parametrize("value", [1., -1., float("nan")])
def test_rejects_additional_metastable_decay(tmp_path, value):
    with pytest.raises(ValueError, match="additional metastable decay"):
        _read_radiative_calibration(*_write_inputs(tmp_path, metastable_extra=value))


@pytest.mark.parametrize(("inputs", "message"), [
    ({"self_decay": 1.}, "self-transition"),
    ({"ground2": 0.}, "intercombination decay rate"),
])
def test_rejects_invalid_triplet_decay(tmp_path, inputs, message):
    with pytest.raises(ValueError, match=message):
        _read_radiative_calibration(*_write_inputs(tmp_path, **inputs))


def test_radiative_cascade_redistributes_to_correct_destination(tmp_path):
    result = _read_radiative_calibration(*_write_inputs(tmp_path, cascade=100.))
    # P2 has 1/200 probability of direct J=2 emission and 100/200
    # probability of reaching P1, which emits J=1 with probability 3/4.
    assert result.branching[2] == pytest.approx(1 / 200 + .5 * .75)
    weighted_j1 = .75 / 3 + (.5 * .75) * 5 / 9
    weighted_j2 = (1 / 200) * 5 / 9
    assert result.redistribution == pytest.approx(
        (weighted_j1 / (weighted_j1 + weighted_j2),
         weighted_j2 / (weighted_j1 + weighted_j2))
    )


def test_rejects_mismatched_ions(tmp_path):
    level, radiative = _write_inputs(tmp_path)
    with fits.open(radiative, mode="update") as hdus:
        hdus[1].header["ELEMENT"] = 11
    with pytest.raises(ValueError, match="different ions"):
        _read_radiative_calibration(level, radiative)


def test_accepts_missing_element_only_with_matching_ion_name(tmp_path):
    level, radiative = _write_inputs(tmp_path)
    with fits.open(radiative, mode="update") as hdus:
        del hdus[1].header["ELEMENT"]
    assert _read_radiative_calibration(level, radiative).element == 8
    with fits.open(radiative, mode="update") as hdus:
        hdus[1].header["ION_NAME"] = "Ne IX"
    with pytest.raises(ValueError, match="different ions"):
        _read_radiative_calibration(level, radiative)
