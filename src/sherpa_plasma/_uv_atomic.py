"""Extract He-like atomic inputs from an installed AtomDB provider."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class _RadiativeCalibration:
    element: int
    ion: int
    ground: int
    metastable: int
    triplet_p: tuple[int, int, int]
    forbidden_rate: float
    branching: tuple[float, float, float]
    effective_branching: float
    redistribution: tuple[float, float]
    omitted_decay_fraction: tuple[float, float, float]


def _read_radiative_calibration(level_file, radiative_file):
    """Derive the statistical-term radiative calibration, without downloads.

    Level indices are one-based AtomDB indices. Radiative cascades among
    the triplet P levels are included. Other decay channels are permitted
    only below 1e-12 of each level's modeled rate and are reported separately.
    """
    from astropy.io import fits

    with fits.open(level_file, memmap=False) as levels_hdu:
        header = levels_hdu[1].header
        element, charge = header.get("ELEMENT"), header.get("ION_STAT")
        if (not isinstance(element, int) or not isinstance(charge, int)
                or element < 3 or charge != element - 2):
            raise ValueError("level file must identify a He-like ion")
        level_name = str(header.get("ION_NAME", "")).strip()
        levels = levels_hdu[1].data

        def find_level(config, n, orbital, spin, degeneracy):
            selection = (
                (np.char.strip(np.asarray(levels["ELEC_CONFIG"], dtype=str)) == config)
                & (levels["N_QUAN"] == n)
                & (levels["L_QUAN"] == orbital)
                & (levels["S_QUAN"] == spin)
                & (levels["LEV_DEG"] == degeneracy)
            )
            found = np.flatnonzero(selection)
            if len(found) != 1:
                raise ValueError(f"expected exactly one level for {config}, "
                                 f"n={n}, L={orbital}, S={spin}, g={degeneracy}")
            return int(found[0]) + 1

        ground = find_level("1s2", 1, 0, 0, 1)
        metastable = find_level("1s1 2s1", 2, 0, 1, 3)
        triplet_p = tuple(find_level("1s1 2p1", 2, 1, 1, g) for g in (1, 3, 5))
        p_energies = np.asarray(levels["ENERGY"], dtype=float)[
            np.asarray(triplet_p) - 1
        ]
        if not np.isfinite(p_energies).all():
            raise ValueError("triplet P energies must be finite")

    with fits.open(radiative_file, memmap=False) as radiative_hdu:
        header = radiative_hdu[1].header
        radiative_element = header.get("ELEMENT")
        radiative_name = str(header.get("ION_NAME", "")).strip()
        if (header.get("ION_STAT") != charge
                or (radiative_element is not None and radiative_element != element)
                or (radiative_element is None
                    and (not level_name or level_name != radiative_name))
                or (level_name and radiative_name and level_name != radiative_name)):
            raise ValueError("level and radiative files identify different ions")
        rows = radiative_hdu[1].data

        def rate(upper, lower, *, required):
            matched = rows[(rows["UPPER_LEV"] == upper) & (rows["LOWER_LEV"] == lower)]
            if len(matched) > 1 or (required and len(matched) != 1):
                raise ValueError(f"expected {'one' if required else 'at most one'} "
                                 f"radiative transition {upper}->{lower}")
            if not len(matched):
                return 0.0
            result = float(matched["EINSTEIN_A"][0])
            if not math.isfinite(result) or result < 0:
                raise ValueError(f"invalid radiative rate for transition {upper}->{lower}")
            return result

        forbidden = rate(metastable, ground, required=True)
        if forbidden <= 0:
            raise ValueError("forbidden transition rate must be positive")
        metastable_other = rows[(rows["UPPER_LEV"] == metastable)
                                & (rows["LOWER_LEV"] != ground)]
        if len(metastable_other):
            rates = np.asarray(metastable_other["EINSTEIN_A"], dtype=float)
            if not np.isfinite(rates).all() or np.any(rates < 0):
                raise ValueError("invalid additional metastable decay")
            if np.any(rates > 0):
                raise ValueError("unsupported additional metastable decay")
        direct = np.zeros((3, 2))
        cascade = np.zeros((3, 3))
        total = np.zeros(3)
        omitted = []
        for j, upper in enumerate(triplet_p):
            ground_rate = rate(upper, ground, required=j != 0)
            metastable_rate = rate(upper, metastable, required=True)
            if metastable_rate <= 0:
                raise ValueError(f"metastable decay rate for J={j} must be positive")
            if j == 0 and ground_rate != 0:
                raise ValueError("J=0 ground decay is not supported by the triplet model")
            if j > 0 and ground_rate <= 0:
                raise ValueError(f"intercombination decay rate for J={j} must be positive")
            if rate(upper, upper, required=False) != 0:
                raise ValueError(f"invalid self-transition from J={j}")
            if j > 0:
                direct[j, j - 1] = ground_rate
            for other_j, other_upper in enumerate(triplet_p):
                if other_j != j:
                    cascade[j, other_j] = rate(upper, other_upper, required=False)
                    if cascade[j, other_j] and p_energies[j] <= p_energies[other_j]:
                        raise ValueError("radiative cascade must decrease level energy")
            modeled_total = ground_rate + metastable_rate + float(cascade[j].sum())
            total[j] = modeled_total
            other = rows[(rows["UPPER_LEV"] == upper)
                         & (rows["LOWER_LEV"] != ground)
                         & (rows["LOWER_LEV"] != metastable)
                         & ~np.isin(rows["LOWER_LEV"], triplet_p)]
            if len(other):
                other_rates = np.asarray(other["EINSTEIN_A"], dtype=float)
                if not np.isfinite(other_rates).all() or np.any(other_rates < 0):
                    raise ValueError(f"invalid additional radiative decay from J={j}")
                other_total = float(np.sum(other_rates))
                if not math.isfinite(other_total):
                    raise ValueError(f"invalid additional radiative decay from J={j}")
                fraction = other_total / modeled_total
                if fraction > 1.e-12:
                    raise ValueError(f"unsupported additional radiative decay from J={j}")
            else:
                fraction = 0.0
            omitted.append(fraction)
    try:
        destination = np.linalg.solve(np.diag(total) - cascade, direct)
    except np.linalg.LinAlgError as exc:
        raise ValueError("invalid radiative cascade network") from exc
    if (not np.isfinite(destination).all() or np.any(destination < 0)
            or np.any(destination.sum(axis=1) > 1 + 1.e-12)):
        raise ValueError("invalid radiative cascade probabilities")
    branching = destination.sum(axis=1)
    weights = np.array((1 / 9, 3 / 9, 5 / 9))
    ground_weights = weights @ destination
    effective = float(ground_weights.sum())
    if effective <= 0:
        raise ValueError("effective branching probability must be positive")
    redistribution = tuple(float(value / effective) for value in ground_weights)
    return _RadiativeCalibration(element, charge + 1, ground, metastable, triplet_p, forbidden,
                                 tuple(float(value) for value in branching), effective, redistribution,
                                 tuple(omitted))
