"""Resolve the atomic data for He-like triplet pumping from an AtomDB filemap."""

from dataclasses import dataclass
from pathlib import Path

from ._uv_atomic import _read_radiative_calibration
from .plasma import AtomDB, LineID, _element


@dataclass(frozen=True)
class AtomDBTripletProvider:
    """Read triplet calibrations from an existing local AtomDB connection."""

    db: AtomDB

    def __post_init__(self):
        if not isinstance(self.db, AtomDB):
            raise TypeError("the triplet provider requires an AtomDB connection")

    def atomic_files(self, element):
        z = _element(element)
        paths = {}
        with self.db.filemap.open(encoding="utf-8") as file:
            for line in file:
                parts = line.split()
                if len(parts) != 4:
                    continue
                try:
                    kind, number, ion = map(int, parts[:3])
                except ValueError:
                    continue
                if kind not in (2, 3) or number != z or ion != z - 1:
                    continue
                if kind in paths:
                    raise ValueError(f"duplicate AtomDB filemap entry for element {z}, type {kind}")
                path = self.db._path(parts[3])
                if not path.is_file():
                    compressed = Path(str(path) + ".gz")
                    if not compressed.is_file():
                        raise FileNotFoundError(f"AtomDB filemap references missing atomic file: {path}")
                    path = compressed
                paths[kind] = path.resolve()
        if set(paths) != {2, 3}:
            raise FileNotFoundError(f"AtomDB filemap lacks LV/LA files for element {z}, ion {z - 1}")
        return paths[2], paths[3]

    def calibration(self, element):
        z = _element(element)
        return self._calibration(z, self.atomic_files(z))

    @staticmethod
    def _calibration(z, files):
        result = _read_radiative_calibration(*files)
        if result.element != z or result.ion != z - 1:
            raise ValueError("AtomDB filemap points to atomic data for a different ion")
        return result

    def line_ids(self, element):
        calibration = self.calibration(element)
        upper = (calibration.metastable, *calibration.triplet_p[1:])
        return tuple(LineID(calibration.element, calibration.ion, index, calibration.ground)
                     for index in upper)
