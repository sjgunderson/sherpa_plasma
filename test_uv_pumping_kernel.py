"""Tests for the dimensionless He-like photon-transfer calculation."""

from decimal import Decimal, localcontext
import random

import pytest

from sherpa_plasma._uv_pumping import _transfer_triplet_photons


def _reference_transfer(f, i1, i2, p):
    with localcontext() as context:
        context.prec = 100
        f, i1, i2, p = (Decimal.from_float(float(value))
                         for value in (f, i1, i2, p))
        i = i1 + i2
        if not f or not i or not p:
            return 0.0
        return float(f * p * i / (f + i + p * i))


@pytest.mark.parametrize("f,i1,i2,p", [
    (3.2718461694, .8, .2, 0.),
    (3.2718461694, .8, .2, 1.),
    (3.2718461694, .8, .2, 1.e-300),
    (3.2718461694, .8, .2, 1.e300),
    (0., 1., 2., 4.),
    (0., 0., 0., 4.),
    (1., 0., 0., 4.),
    (1.e300, 1.e200, 1.e199, 1.e-300),
    (1.e300, 1.e300, 1.e300, 1.e300),
    (1.e-300, 1.e-300, 1.e-300, 1.e300),
    (1.e300, 1.e-300, 0., 1.e300),
    (1.e308, 5.e-324, 5.e-324, 1.e308),
    (1.e308, 1.e308, 1.e308, 5.e-324),
])
def test_transfer_matches_high_precision_reference(f, i1, i2, p):
    actual = _transfer_triplet_photons(f, i1, i2, p)
    assert actual == pytest.approx(_reference_transfer(f, i1, i2, p),
                                   rel=1.e-12, abs=5.e-324)
    assert 0 <= actual <= f


@pytest.mark.parametrize("p", [0., 1.e-6, 1., 100., 1.e12])
def test_transfer_conserves_photons_and_matches_target_ratio(p):
    f, i1, i2 = 3.2718461694, .8, .2
    delta = _transfer_triplet_photons(f, i1, i2, p)
    assert (f - delta) + (i1 + i2 + delta) == pytest.approx(f + i1 + i2)
    assert (f - delta) / (i1 + i2 + delta) == pytest.approx(
        (f / (i1 + i2)) / (1 + p), rel=1.e-14, abs=1.e-16)


def test_integrated_ratio_differs_from_temperature_resolved_ratio():
    triplets = [(9., 1., 0.), (1., 9., 0.)]
    f, i1, i2 = (sum(row[index] for row in triplets) for index in range(3))
    integrated = _transfer_triplet_photons(f, i1, i2, 1.)
    resolved = sum(_transfer_triplet_photons(*row, 1.) for row in triplets)
    assert (f - integrated) / (i1 + i2 + integrated) == pytest.approx(.5)
    assert resolved != pytest.approx(integrated)


def test_transfer_across_wide_dynamic_range():
    rng = random.Random(217)
    for _ in range(500):
        values = [10.0 ** rng.uniform(-300, 300) for _ in range(4)]
        actual = _transfer_triplet_photons(*values)
        expected = _reference_transfer(*values)
        assert actual == pytest.approx(expected, rel=1.e-12, abs=5.e-324)


@pytest.mark.parametrize("values,match", [
    ((-1, 1, 1, 1), "nonnegative"),
    ((1, -1, 1, 1), "nonnegative"),
    ((1, 1, 1, -1), "nonnegative"),
    ((1, 1, 1, float("inf")), "finite"),
    ((float("nan"), 1, 1, 1), "finite"),
])
def test_invalid_inputs_fail_explicitly(values, match):
    with pytest.raises(ValueError, match=match):
        _transfer_triplet_photons(*values)
