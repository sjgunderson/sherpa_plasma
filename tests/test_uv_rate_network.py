"""Verify photon transfer against a separately solved rate network."""

import numpy as np
import pytest

from sherpa_plasma._uv_pumping import _transfer_triplet_photons


@pytest.mark.parametrize("pumping", [0., .01, 1., 100., 1.e6])
def test_transfer_matches_steady_state_level_populations(pumping):
    # States are metastable S, then P0, P1, P2. P2 -> P1 is substantial.
    forbidden = 10.
    ground = np.array([0., 300., 1.])
    metastable = np.array([100., 100., 99.])
    cascade = np.zeros((3, 3))
    cascade[2, 1] = 100.
    weights = np.array([1., 3., 5.]) / 9
    sources = np.array([4., 1., 2., 3.])

    def solve(rate):
        matrix = np.zeros((4, 4))
        matrix[0, 0] = forbidden + rate
        matrix[0, 1:] = -metastable
        matrix[1:, 0] = -rate * weights
        matrix[1:, 1:] = np.diag(ground + metastable + cascade.sum(axis=1)) - cascade.T
        populations = np.linalg.solve(matrix, sources)
        return forbidden * populations[0], ground[1:] * populations[2:]

    unpumped_f, unpumped_i = solve(0.)
    pumped_f, pumped_i = solve(pumping)
    matrix = np.diag(ground + metastable + cascade.sum(axis=1)) - cascade
    direct = np.array([[0., 0.], [ground[1], 0.], [0., ground[2]]])
    destinations = np.linalg.solve(matrix, direct)
    weighted = weights @ destinations
    branching = weighted.sum()
    r0 = unpumped_f / unpumped_i.sum()
    critical = forbidden / (branching * (1 + r0))
    transfer = _transfer_triplet_photons(unpumped_f, *unpumped_i, pumping / critical)

    assert pumped_f == pytest.approx(unpumped_f - transfer, rel=1.e-10, abs=1.e-12)
    np.testing.assert_allclose(pumped_i, unpumped_i + transfer * weighted / branching,
                               rtol=1.e-10, atol=1.e-12)
