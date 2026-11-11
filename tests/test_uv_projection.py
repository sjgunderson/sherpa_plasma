import numpy as np
import pytest

from sherpa_plasma._uv_projection import _unbroadened_line_bins


def test_unbroadened_projection_handles_redshift_and_off_grid():
    edges = np.array([0.5, 0.6, 0.7, 0.9])
    wavelengths = np.array([12.39841974 / .54, 12.39841974 / .75,
                            12.39841974 / .45])
    np.testing.assert_array_equal(
        _unbroadened_line_bins(edges[:-1], edges[1:], wavelengths, 0), [0, 2, -1]
    )
    np.testing.assert_array_equal(
        _unbroadened_line_bins(edges[:-1], edges[1:], wavelengths, .1), [-1, 1, -1]
    )


def test_rejects_unvalidated_near_edge_projection():
    edges = np.array([0.5, 0.6, 0.7])
    wavelength = 12.39841974 / .6
    with pytest.raises(ValueError, match="too close to a bin edge"):
        _unbroadened_line_bins(edges[:-1], edges[1:], [wavelength], 0)


@pytest.mark.parametrize("lo,hi", [
    ([.5, .61], [.6, .7]),
    ([.5, .6], [.6, .5]),
    ([.5], [.6]),
    ([0, .6], [.6, .7]),
])
def test_rejects_unsupported_grids(lo, hi):
    with pytest.raises(ValueError, match="contiguous ordered"):
        _unbroadened_line_bins(lo, hi, [20.], 0)
