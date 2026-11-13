"""Coarse cross-check against independently published radiative rates."""

import json
from pathlib import Path

import pytest


# Porquet & Dubau (2000), Table 2, rows (lower, upper) 1,2; 1,4;
# 1,5; 2,4; 2,5. Their historical values differ from current AtomDB rates.
PUBLISHED_RATES = {
    6: (49.6, 2.159e7, 2.650e4, 5.655e7, 5.735e7),
    7: (253., 1.100e8, 1.030e5, 6.794e7, 6.955e7),
    8: (1060., 4.447e8, 3.330e5, 7.956e7, 8.249e7),
    10: (11000., 4.470e9, 2.270e6, 1.039e8, 1.118e8),
    12: (73300., 2.867e10, 1.060e7, 1.304e8, 1.486e8),
    14: (361000., 1.345e11, 3.890e7, 1.602e8, 1.977e8),
}


@pytest.mark.parametrize("version", ["3_0_9", "3_1_3"])
@pytest.mark.parametrize("element", sorted(PUBLISHED_RATES))
def test_common_ion_calibration_is_consistent_with_literature(version, element):
    from sherpa_plasma import __file__ as package_file

    path = Path(package_file).parent / f"_uv_atomdb_{version}.json"
    with path.open(encoding="utf-8") as file:
        record = json.load(file)["ions"][str(element)]["calibration"]
    forbidden, ground1, ground2, metastable1, metastable2 = PUBLISHED_RATES[element]
    b1 = ground1 / (ground1 + metastable1)
    b2 = ground2 / (ground2 + metastable2)
    published_b = b1 / 3 + 5 * b2 / 9

    # This is an independent order-of-magnitude check, not a precision
    # validation; the two sources use different generations of atomic data.
    assert record["forbidden_rate"] == pytest.approx(forbidden, rel=.25)
    assert record["effective_branching"] == pytest.approx(published_b, rel=.25)
    assert record["forbidden_rate"] / record["effective_branching"] == pytest.approx(
        forbidden / published_b, rel=.25
    )
