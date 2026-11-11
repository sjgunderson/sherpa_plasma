import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def coverage_helper():
    path = Path(__file__).parents[1] / "src" / "sherpa_plasma" / "_uv_model.py"
    spec = importlib.util.spec_from_file_location("uv_model", path)
    module = importlib.util.module_from_spec(spec)
    pytest.importorskip("sherpa")
    spec.loader.exec_module(module)
    return module._dem_active_weights


def test_positive_out_of_range_temperature_is_not_silently_dropped(coverage_helper):
    record = {"temperature_min_keV": 0.2, "temperature_max_keV": 6.46}
    with pytest.raises(ValueError, match="Si temperature is outside"):
        coverage_helper(np.array([1.0, 7.0]), np.array([1.0, 0.1]), record, "Si")


def test_zero_weight_out_of_range_temperature_is_inactive(coverage_helper):
    record = {"temperature_min_keV": 0.2, "temperature_max_keV": 6.46}
    active = coverage_helper(np.array([1.0, 7.0]), np.array([1.0, 0.0]), record, "Si")
    np.testing.assert_array_equal(active, [True, False])
