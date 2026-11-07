"""Local AtomDB/APED browsing and line diagnostics using PyAtomDB.

No Sherpa or PyAtomDB import occurs until the corresponding functionality
is used. Atomic searches need a filemap and LA/LV files; emissivities also need
the APEC line and continuum emissivity files. See PLASMA.md for conventions.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from numbers import Integral
import os
from pathlib import Path
import re
import sys
import tempfile
from threading import RLock

import numpy as np

from .units import HC_KEV_ANGSTROM, convert_bins

_LOCK = RLock()  # PyAtomDB uses process-wide environment variables.
_SYMBOLS = ("", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")
_LINE_DTYPE = np.dtype([
    ("Element", "i4"), ("Ion", "i4"), ("UpperLev", "i4"), ("LowerLev", "i4"),
    ("Lambda", "f8"), ("Energy", "f8"), ("Epsilon", "f8"),
    ("ModelContribution", "f8"), ("Einstein_A", "f8")])
_XSPEC_ABUNDANCE_SETS = {
    "angr": "AG89",
    "aspl": "Asplund09",
    "feld": "Feldman",
    "aneb": "AE82",
    "grsa": "GS98",
    "wilm": "Wilm00",
    "lodd": "Lodd03",
}
_SYMBOL_TO_Z = {symbol.lower(): z for z, symbol in enumerate(_SYMBOLS) if symbol}


def _integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return int(value)


def _element(value):
    if isinstance(value, str):
        try:
            z = _SYMBOLS.index(value.capitalize())
            if z == 0:
                raise ValueError
            return z
        except ValueError:
            raise ValueError(f"unknown AtomDB element {value!r}; use H through Zn") from None
    return _integer(value, "element", 1, 30)


def _elements(values):
    if values is None:
        return tuple(range(1, 31))
    if np.ndim(values) == 0:
        values = (values,)
    result = tuple(_element(v) for v in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("provide distinct elements")
    return result


def _positive(values, name, *, zero=False):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0 if zero else values <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return values


def _temperature(values, teunit):
    values = _positive(values, "temperature")
    if values.ndim > 1 or values.size == 0:
        raise ValueError("temperature must be a scalar or nonempty 1D array")
    if teunit.lower() == "kev":
        return values
    if teunit.lower() == "k":
        return values * (1.380649e-23 / 1.602176634e-16)
    raise ValueError("teunit must be 'keV' or 'K'")


@contextmanager
def _environment(root):
    """Restore even variables changed internally by read_filemap on failure."""
    with _LOCK:
        before = {key: os.environ.get(key) for key in ("ATOMDB", "OLDATOMDB")}
        os.environ["ATOMDB"] = str(root)
        try:
            yield
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@dataclass(frozen=True)
class LineID:
    """Transition identity within one database version; levels are one-based.

    Ion is the spectroscopic stage (charge + 1): O VII is (8, 7).
    Level numbers and wavelengths may change between database versions.
    """

    element: int
    ion: int
    upper: int
    lower: int

    def __post_init__(self):
        z = _element(self.element)
        object.__setattr__(self, "element", z)
        for name, hi in (("ion", z + 1), ("upper", 2**31 - 1), ("lower", 2**31 - 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1, hi))

    @classmethod
    def from_row(cls, row):
        return cls(*(int(row[key]) for key in ("Element", "Ion", "UpperLev", "LowerLev")))

    def mask(self, rows):
        return ((rows["Element"] == self.element) & (rows["Ion"] == self.ion) &
                (rows["UpperLev"] == self.upper) & (rows["LowerLev"] == self.lower))


@dataclass(frozen=True)
class ComponentLineContributions:
    """Weighted line contributions for one captured plasma component."""

    name: str
    model_type: str
    values: np.ndarray

    def __post_init__(self):
        values = np.array(self.values, dtype=float, copy=True)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("component line contributions must be finite and 1D")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class PlasmaLineContributions:
    """Line metadata and weighted contributions from captured components.

    ``lines`` uses the same fields as :meth:`AtomDB.find_lines`; its ``Epsilon``
    field is NaN because the weighted values are separate from single-
    temperature emissivities. ``ModelContribution`` matches ``total``.
    """

    lines: np.ndarray
    components: tuple[ComponentLineContributions, ...]
    total: np.ndarray

    def __post_init__(self):
        lines = np.array(self.lines, dtype=_LINE_DTYPE, copy=True)
        total = np.array(self.total, dtype=float, copy=True)
        components = tuple(self.components)
        if lines.ndim != 1 or total.ndim != 1 or len(lines) != len(total):
            raise ValueError("line metadata and total contributions must be matching 1D arrays")
        if not np.isfinite(total).all():
            raise ValueError("total line contributions must be finite")
        if any(len(component.values) != len(lines) for component in components):
            raise ValueError("component contributions must match the line count")
        lines.setflags(write=False)
        total.setflags(write=False)
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "total", total)


def _line_ids(lines):
    if lines is None:
        return None
    if isinstance(lines, LineID):
        return (lines,)
    return tuple(row if isinstance(row, LineID) else LineID.from_row(row) for row in lines)


def _xspec_abundance_set(table):
    if not isinstance(table, str) or not table.strip():
        raise ValueError("captured component has no XSPEC abundance table")
    key = table.split(":", 1)[0].strip().lower()
    try:
        return _XSPEC_ABUNDANCE_SETS[key]
    except KeyError:
        raise ValueError(f"unsupported XSPEC abundance table {table!r}") from None


def _component_abundances(component, *, allow_missing=False):
    """Return AtomDB multipliers and elements with zero effective abundance."""
    parameters = {}
    for name, value in component.parameters.items():
        key = str(name).lower()
        if key in ("abundanc", "abund") or key in _SYMBOL_TO_Z:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid abundance parameter {name!r} in {component.name}")
            parameters[key] = value

    scalar = parameters.get("abundanc", parameters.get("abund"))
    explicit = {
        key: value for key, value in parameters.items()
        if key in _SYMBOL_TO_Z
    }
    if scalar is None and not explicit:
        if allow_missing:
            return {}, set()
        raise ValueError(f"no abundance parameters captured for {component.name}")

    abundances = {}
    if scalar is not None:
        abundances.update({
            symbol: scalar
            for z, symbol in enumerate(_SYMBOLS)
            if z > 2 and symbol
        })
    for key, value in explicit.items():
        abundances[_SYMBOLS[_SYMBOL_TO_Z[key]]] = value
    suppressed = {
        _SYMBOL_TO_Z[symbol.lower()]
        for symbol, value in abundances.items()
        if value == 0
    }
    return abundances, suppressed


def _canonical(rows, *, element=None, ion=None):
    result = np.empty(len(rows), dtype=_LINE_DTYPE)
    for key in result.dtype.names:
        result[key] = np.nan if result.dtype[key].kind == "f" else 0
    names = {key.lower(): key for key in rows.dtype.names}
    for key in result.dtype.names:
        if key.lower() in names and key.lower() != "modelcontribution":
            result[key] = rows[names[key.lower()]]
    if element is not None:
        result["Element"] = element
        result["Ion"] = ion
    with np.errstate(divide="ignore", invalid="ignore"):
        result["Energy"] = HC_KEV_ANGSTROM / result["Lambda"]
    return result[np.isfinite(result["Lambda"]) & (result["Lambda"] > 0)]


class AtomDB:
    """A connection to an existing local APED/AtomDB installation.

    ``root`` defaults to $ATOMDB. File names may be absolute or relative
    to root; versioned names work without changing global symlinks.
    ``abundset=None`` retains the emissivity files' native abundance set.
    Alternate sets require the local ABUND file referenced by filemap.
    The connection is serialized and may be used as a context manager.
    Emissivity tables load lazily and remain open until close().
    This interface does not construct or evaluate fitting components.
    """

    def __init__(self, root=None, *, linefile="apec_line.fits", cocofile="apec_coco.fits",
                 nei_linefile="apec_nei_line.fits",
                 nei_cocofile="apec_nei_comp.fits", filemap="filemap",
                 abundset=None):
        root = root if root is not None else os.environ.get("ATOMDB")
        if not root:
            raise ValueError("provide the local AtomDB root or set ATOMDB")
        self.root = Path(os.path.expandvars(str(root))).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"AtomDB directory does not exist: {self.root}")
        self.linefile, self.cocofile, self.filemap = (
            self._path(name) for name in (linefile, cocofile, filemap))
        self.nei_linefile, self.nei_cocofile = (
            self._path(name) for name in (nei_linefile, nei_cocofile))
        self.abundset = abundset
        self._api = None
        self._session = None
        self._nei_sessions = {}
        self._sherpa_eigen_path = None
        self._sherpa_eigen_hdul = None
        self._datacache = {}
        self._filemap = None
        self._scratch = None
        self._closed = False

    def _path(self, name):
        text = str(name).replace("${ATOMDB}", str(self.root)).replace("$ATOMDB", str(self.root))
        path = Path(os.path.expandvars(text)).expanduser()
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _require(path):
        if not path.is_file():
            raise FileNotFoundError(f"required local AtomDB file is missing: {path}")

    @contextmanager
    def _using(self, *, cache=False):
        with _LOCK:
            if self._closed:
                raise RuntimeError("this AtomDB connection is closed")
            if self._api is None:
                # PyAtomDB's import checks for userdata. Bootstrap only its
                # import against our own directory; never initialize/download
                # into the user's existing installation.
                self._scratch = tempfile.TemporaryDirectory(prefix="sherpa-isis-atomdb-")
                Path(self._scratch.name, "userdata").touch()
                try:
                    with _environment(self._scratch.name):
                        self._api = importlib.import_module("pyatomdb")
                except ImportError as exc:
                    self._scratch.cleanup()
                    self._scratch = None
                    raise ImportError("install the plasma extra: pip install 'sherpa_plasma[plasma]'") from exc
            with _environment(self._scratch.name if cache else self.root):
                yield self._api

    def _get_data(self, element, ion, kind, *, required=True, datacache=None):
        self._require(self.filemap)
        with self._using() as api:
            data = api.atomdb.get_data(element, ion, kind, offline=True,
                datacache=self._datacache if datacache is None else datacache,
                # PyAtomDB uses lowercase for atomic files but historical
                # FileMap spelling for miscellaneous files such as ABUND.
                settings={"filemap": str(self.filemap), "FileMap": str(self.filemap),
                          "atomdbroot": str(self.root)})
            if data is False or data is None:
                if required:
                    raise FileNotFoundError(f"local AtomDB has no {kind} data for element={element}, ion={ion}")
                return None
            return data

    def _cie(self):
        if self._session is None:
            self._require(self.linefile)
            self._require(self.cocofile)
            with self._using(cache=True) as api:
                # Explicit files and abundset=False avoid implicit downloads.
                # PyAtomDB's generated pickle cache stays in our scratch dir.
                s = api.spectrum.CIESession(linefile=str(self.linefile), cocofile=str(self.cocofile),
                                           abundset=False)
                s.set_eebrems(False)  # table-based model; no extra ion-balance calculation
            try:
                if self.abundset is not None:
                    self._set_abundset(s, self.abundset)
            except Exception:
                s.linedata.close()
                s.cocodata.close()
                raise
            self._session = s
        return self._session

    def _nei(self, *, pshock=False):
        kind = "pshock" if pshock else "nei"
        if kind not in self._nei_sessions:
            self._require(self.nei_linefile)
            self._require(self.nei_cocofile)
            with self._using(cache=True) as api:
                session_type = api.spectrum.PShockSession if pshock else api.spectrum.NEISession
                session = session_type(
                    linefile=str(self.nei_linefile),
                    cocofile=str(self.nei_cocofile),
                    abundset=False,
                )
                session.set_eebrems(False)
            try:
                self._preload_nei_eigen(session)
                if self.abundset is not None:
                    self._set_abundset(session, self.abundset)
            except Exception:
                session.linedata.close()
                session.cocodata.close()
                raise
            self._nei_sessions[kind] = session
        return self._nei_sessions[kind]

    @staticmethod
    def _eigen_version_key(path):
        match = re.fullmatch(
            r"eigen_v([0-9]+(?:\.[0-9]+)*)\.fits(?:\.gz)?",
            path.name,
            flags=re.IGNORECASE,
        )
        return tuple(int(part) for part in match.group(1).split(".")) if match else ()

    def _find_sherpa_eigen_file(self):
        directories = []
        headas = os.environ.get("HEADAS")
        if headas:
            directories.append(
                Path(headas).expanduser().parent / "spectral" / "modelData"
            )
        directories.append(Path(sys.prefix) / "spectral" / "modelData")

        candidates = set()
        for directory in directories:
            try:
                candidates.update(directory.glob("eigen_v*.fits"))
                candidates.update(directory.glob("eigen_v*.fits.gz"))
            except OSError:
                continue
        candidates = [
            path for path in candidates
            if self._eigen_version_key(path)
        ]
        if not candidates:
            return None
        return max(candidates, key=self._eigen_version_key).resolve()

    def _expected_local_eigen_file(self, element):
        version_file = self.root / "VERSION"
        version = version_file.read_text().strip() if version_file.is_file() else ""
        if version in ("2.0.0", "2.0.1", "2.0.2", "3.0.0", "3.0.1",
                       "3.0.2", "3.0.3"):
            eigen_version = "3.0"
        elif version in ("3.0.4", "3.0.5", "3.0.6"):
            eigen_version = "3.0.4"
        elif version in ("3.0.7", "3.0.8", "3.0.9"):
            eigen_version = "3.0.7"
        elif version.startswith("3.1"):
            eigen_version = "3.1.0"
        else:
            eigen_version = "VERSION-dependent"
        symbol = _SYMBOLS[element].lower()
        suffix = f"eigen{symbol}_v{eigen_version}.fits"
        return self.root / "APED" / "ionbal" / "eigen" / suffix

    def _load_local_eigen(self, session, element):
        cache = {}
        try:
            with self._using() as api:
                data = api.atomdb.get_data(
                    element, False, "EIGEN", offline=True,
                    datacache=cache,
                    settings={
                        "filemap": str(self.filemap),
                        "FileMap": str(self.filemap),
                        "atomdbroot": str(self.root),
                    },
                )
        except KeyError as exc:
            expected = self._expected_local_eigen_file(element)
            raise FileNotFoundError(
                f"local AtomDB has no EIGEN data for {_SYMBOLS[element]} "
                f"(Z={element}); expected a file such as {expected} or an "
                "EIGEN entry in the configured filemap"
            ) from exc
        if data is False or data is None:
            expected = self._expected_local_eigen_file(element)
            raise FileNotFoundError(
                f"local AtomDB has no EIGEN data for {_SYMBOLS[element]} "
                f"(Z={element}); expected a file such as {expected} or an "
                "EIGEN entry in the configured filemap"
            )

        target_data = session.datacache.setdefault("data", {})
        target_misc = target_data.setdefault("misc", {})
        target_eigen = target_misc.setdefault("EIGEN", {})
        target_eigen[element] = data
        source_sums = cache.get("datasums", {}).get("misc", {}).get("EIGEN", {})
        target_sums = (
            session.datacache.setdefault("datasums", {})
            .setdefault("misc", {}).setdefault("EIGEN", {})
        )
        if element in source_sums:
            target_sums[element] = source_sums[element]

    def _preload_nei_eigen(self, session):
        eigen_path = self._find_sherpa_eigen_file()
        if eigen_path is not None:
            if (self._sherpa_eigen_hdul is None
                    or eigen_path != self._sherpa_eigen_path):
                from astropy.io import fits
                self._sherpa_eigen_hdul = fits.open(eigen_path, memmap=False)
                self._sherpa_eigen_path = eigen_path

            hdu_names = {
                str(hdu.name).lower()
                for hdu in self._sherpa_eigen_hdul
            }
            target_data = session.datacache.setdefault("data", {})
            target_misc = target_data.setdefault("misc", {})
            target_eigen = target_misc.setdefault("EIGEN", {})
            target_sums = (
                session.datacache.setdefault("datasums", {})
                .setdefault("misc", {}).setdefault("EIGEN", {})
            )
            for element in session.elements:
                symbol = _SYMBOLS[int(element)].lower()
                if symbol in hdu_names:
                    target_eigen[int(element)] = self._sherpa_eigen_hdul
                    target_sums[int(element)] = self._sherpa_eigen_hdul[0].header.get(
                        "DATASUM", ""
                    )
                else:
                    self._load_local_eigen(session, int(element))
            return

        # Without a Sherpa bundle, use PyAtomDB's configured local filemap.
        # Isolated temporary caches avoid PyAtomDB's EIGEN miss-path KeyError
        # corrupting the shared per-element cache.
        for element in session.elements:
            self._load_local_eigen(session, int(element))

    @staticmethod
    def _check_nei_temperature(session, temperature):
        grid = np.asarray(session.spectra.kTlist, dtype=float)
        if np.any(temperature < grid[0]) or np.any(temperature > grid[-1]):
            raise ValueError(
                f"temperature outside local NEI emissivity range "
                f"[{grid[0]:g}, {grid[-1]:g}] keV"
            )
        return session

    def _set_abundset(self, session, abundset):
        if abundset == session.abundset:
            return
        data = self._get_data(False, False, "ABUND", datacache=session.datacache)
        if abundset not in data[1].data["Source"]:
            raise ValueError(f"abundance set absent from local ABUND file: {abundset}")
        with self._using():
            session.set_abundset(abundset)

    def _check_temperature(self, kT):
        s = self._cie()
        grid = np.asarray(s.spectra.kTlist)
        if np.any(kT < grid[0]) or np.any(kT > grid[-1]):
            raise ValueError(f"temperature outside local emissivity range [{grid[0]:g}, {grid[-1]:g}] keV")
        return s

    @staticmethod
    def _tabulated_temperature(session, temperature):
        # Backend log-temperature rounding can produce negative interpolation
        # weights even at an exact table temperature. Interpolating line lists
        # also intersects neighboring transition sets, dropping weak lines that
        # exist only in the selected table. Snap only within float64 roundoff.
        grid = np.asarray(session.spectra.kTlist, dtype=float)
        return bool(np.any(np.isclose(float(temperature), grid,
                                     rtol=8*np.finfo(float).eps, atol=0.)))

    def _configure_session(self, session, abundances, *, abundset=None):
        abundset = self.abundset if abundset is None else abundset
        if abundset is None:
            abundset = session.default_abundset
        self._set_abundset(session, abundset)
        session.set_abund(session.elements, 1.0)
        if abundances is not None:
            seen = set()
            for element, value in abundances.items():
                z = _element(element)
                if z in seen:
                    raise ValueError("duplicate element abundance")
                seen.add(z)
                value = _positive(value, "abundance", zero=True)
                if value.ndim != 0:
                    raise ValueError("each abundance must be scalar")
                session.set_abund(z, float(value))
        return session

    def _configure(self, abundances, *, abundset=None):
        return self._configure_session(
            self._cie(), abundances, abundset=abundset
        )


    @property
    def temperature_range(self):
        """Inclusive minimum/maximum available temperatures, in keV."""
        with self._using():
            values = self._cie().spectra.kTlist
            return float(values[0]), float(values[-1])

    @property
    def temperatures(self):
        """Detached copy of the native CIE temperature knots, in keV."""
        with self._using():
            return np.array(self._cie().spectra.kTlist, dtype=float, copy=True)

    @property
    def info(self):
        """Paths and version information for reproducible analyses."""
        version = self.root / "VERSION"
        result = {"root": str(self.root), "linefile": str(self.linefile.resolve()),
                  "cocofile": str(self.cocofile.resolve()),
                  "nei_linefile": str(self.nei_linefile.resolve()),
                  "nei_cocofile": str(self.nei_cocofile.resolve()),
                  "filemap": str(self.filemap.resolve()),
                  "version": version.read_text().strip() if version.is_file() else None,
                  "abundset": self.abundset}
        if self._session is not None:
            result["abundset"] = self._session.abundset
            result["line_checksum"] = self._session.linedata[0].header.get("CHECKSUM")
            result["continuum_checksum"] = self._session.cocodata[0].header.get("CHECKSUM")
        return result

    def levels(self, element, ion):
        """Return a copy of an ion's raw LV table; level 1 is row 0."""
        z = _element(element)
        ion = _integer(ion, "ion", 1, z + 1)
        with self._using():
            return self._get_data(z, ion, "LV")[1].data.copy()

    def transitions(self, element, ion):
        """Return a copy of an ion's raw LA table, retaining one-based levels."""
        z = _element(element)
        ion = _integer(ion, "ion", 1, z + 1)
        with self._using():
            return self._get_data(z, ion, "LA")[1].data.copy()

    def line_info(self, line):
        """Atomic transition rows and the upper/lower level records."""
        if not isinstance(line, LineID):
            line = LineID.from_row(line)
        rows = self.transitions(line.element, line.ion)
        names = {name.lower(): name for name in rows.dtype.names}
        rows = rows[(rows[names["upper_lev"]] == line.upper) &
                    (rows[names["lower_lev"]] == line.lower)]
        if not len(rows):
            raise LookupError(f"transition absent from local LA data: {line}")
        levels = self.levels(line.element, line.ion)
        # Astropy FITS_record has no copy(). Copy a one-row table first,
        # retaining detached records for both FITS_rec and NumPy inputs.
        return {"line": line, "transitions": rows, "upper": levels[[line.upper - 1]].copy()[0],
                "lower": levels[[line.lower - 1]].copy()[0], "database": self.info}

    def find_lines(self, lo, hi, *, unit="Angstrom", element=None, ion=None,
                   kT=None, teunit="keV", abundances=None, abundset=None,
                   min_emissivity=None, brightest=None, model=None,
                   dataset_id=None, session=None):
        """Search atomic transitions or rank supported plasma lines.

        Return a NumPy structured array. Lambda is Angstrom, Energy keV,
        Epsilon photons cm^3/s, ModelContribution the total AtomDB
        emissivity contribution for a captured model, and Einstein_A s^-1.
        Unavailable fields are
        NaN. Without ``kT`` or ``brightest`` this searches the LA database.
        ``brightest`` without ``kT`` ranks lines from the current supported
        CIE/NEI/PShock Sherpa/XSPEC source by total contribution; ``model`` or
        ``dataset_id`` and ``session`` can select a source explicitly. With
        ``kT`` this searches the single-temperature CIE emissivity tables.
        ``abundances`` supplies per-element multipliers and ``abundset``
        selects their baseline for that explicit-temperature search. Bounds
        are inclusive.
        """
        elo, ehi, _ = convert_bins(np.atleast_1d(lo), np.atleast_1d(hi), unit, "keV")
        if elo.size != 1 or not np.isfinite([lo, hi]).all() or lo <= 0 or hi <= lo:
            raise ValueError("provide finite scalar search bounds with 0 < lo < hi")
        zs = _elements(element)
        ions = None if ion is None else tuple(_integer(i, "ion", 1, 31) for i in np.atleast_1d(ion))
        if min_emissivity is not None:
            min_emissivity = float(_positive(min_emissivity, "min_emissivity", zero=True))
        if brightest is not None:
            brightest = _integer(brightest, "brightest", 1, 2**31 - 1)
        source_selected = model is not None or dataset_id is not None or session is not None
        if kT is None and brightest is not None:
            if abundances is not None or abundset is not None:
                raise ValueError(
                    "abundances and abundset are taken from the captured "
                    "model when ranking model contributions"
                )
            result = self.query_plasma_lines(
                lo, hi, model=model, dataset_id=dataset_id, session=session,
                unit=unit,
            )
            rows = result.lines.copy()
            rows["ModelContribution"] = result.total
            mask = (np.isin(rows["Element"], zs) & (rows["Lambda"] > 0))
            if ions is not None:
                mask &= np.isin(rows["Ion"], ions)
            if min_emissivity is not None:
                mask &= rows["ModelContribution"] >= min_emissivity
            rows = rows[mask]
            order = np.argsort(-rows["ModelContribution"], kind="stable")[:brightest]
            return rows[order]
        if source_selected:
            raise ValueError(
                "model, dataset_id, and session are only valid when ranking "
                "the current model with brightest and without kT"
            )
        if kT is None and (
            min_emissivity is not None or abundances is not None or abundset is not None
        ):
            raise ValueError(
                "emissivity filtering and abundances require kT or "
                "brightest model ranking"
            )
        with self._using() as api:
            if kT is None:
                self._require(self.filemap)
                if self._filemap is None:
                    self._filemap = api.atomdb.read_filemap(str(self.filemap), atomdbroot=str(self.root))
                pieces = []
                for index, (z, stage) in enumerate(zip(self._filemap["Z"], self._filemap["z1"])):
                    if z not in zs or ions is not None and stage not in ions:
                        continue
                    data = self._get_data(int(z), int(stage), "LA", required=False)
                    if data is None:
                        if str(self._filemap["la"][index]).strip():
                            raise FileNotFoundError(f"filemap references unavailable local LA data: element={z}, ion={stage}")
                        continue
                    raw = data[1].data
                    names = {key.lower(): key for key in raw.dtype.names}
                    rows = np.zeros(len(raw), dtype=_LINE_DTYPE)
                    rows["Element"], rows["Ion"] = z, stage
                    rows["UpperLev"], rows["LowerLev"] = raw[names["upper_lev"]], raw[names["lower_lev"]]
                    rows["Lambda"] = raw[names["wavelen"]]
                    if "wave_obs" in names:
                        observed = raw[names["wave_obs"]]
                        valid = np.isfinite(observed) & (observed > 0)
                        rows["Lambda"][valid] = observed[valid]
                    rows["Einstein_A"] = raw[names["einstein_a"]]
                    rows["Epsilon"] = np.nan
                    rows["ModelContribution"] = np.nan
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rows["Energy"] = HC_KEV_ANGSTROM / rows["Lambda"]
                    pieces.append(rows)
                rows = np.concatenate(pieces) if pieces else np.empty(0, dtype=_LINE_DTYPE)
            else:
                temp = _temperature(kT, teunit)
                if temp.ndim != 0:
                    raise ValueError("find_lines requires a scalar temperature")
                s = self._check_temperature(temp)
                self._configure(abundances, abundset=abundset)
                # Use the exact table at every tabulated temperature.
                raw = s.return_linelist(float(temp), [float(elo[0]), float(ehi[0])],
                    specunit="keV", teunit="keV", apply_aeff=False,
                    nearest=self._tabulated_temperature(s, temp))
                rows = _canonical(raw)
            mask = ((rows["Energy"] >= elo[0]) & (rows["Energy"] <= ehi[0]) &
                    np.isin(rows["Element"], zs) & (rows["Lambda"] > 0))
            if ions is not None:
                mask &= np.isin(rows["Ion"], ions)
            if min_emissivity is not None:
                mask &= rows["Epsilon"] >= min_emissivity
            rows = rows[mask].copy()
            order = np.argsort(rows["Lambda"], kind="stable")
            if brightest is not None:
                order = np.argsort(-rows["Epsilon"], kind="stable")[:brightest]
            return rows[order]

    def _line_emissivity_by_line(self, selected, temp, *, abundances=None, abundset=None):
        values = np.zeros((len(selected), temp.size))
        with self._using():
            s = self._check_temperature(temp)
            self._configure(abundances, abundset=abundset)
            for index, t in enumerate(temp.reshape(-1)):
                raw = s.return_linelist(float(t), [np.finfo(float).tiny, 1.e10],
                    specunit="A", teunit="keV", apply_aeff=False,
                    nearest=self._tabulated_temperature(s, t))
                for line_index, line in enumerate(selected):
                    # Interpolating through the public line list also handles
                    # the lowest table temperature's backend edge case.
                    values[line_index, index] = np.sum(raw["Epsilon"][line.mask(raw)])
        return values.reshape((len(selected),) + temp.shape)

    def line_emissivity(self, lines, temperatures, *, teunit="keV", abundances=None,
                        abundset=None):
        """Sum emissivities of selected transitions at scalar/array temperatures.

        Returns photons cm^3/s, including the configured abundance set and
        multipliers. ``abundset`` overrides the connection's abundance
        baseline for this query. Duplicate LineIDs are rejected. Missing lines
        yield zero.
        """
        selected = _line_ids(lines)
        if selected is None or len(set(selected)) != len(selected):
            raise ValueError("provide distinct transition identities")
        temp = _temperature(temperatures, teunit)
        values = self._line_emissivity_by_line(
            selected, temp, abundances=abundances, abundset=abundset
        ).sum(axis=0)
        return float(values) if temp.ndim == 0 else values

    @staticmethod
    def _initial_population(init_pop, teunit):
        if isinstance(init_pop, str):
            if init_pop.lower() not in ("ionizing", "recombining"):
                raise ValueError("init_pop must be 'ionizing', 'recombining', or a temperature")
            return init_pop.lower()
        if isinstance(init_pop, (bool, np.bool_)):
            raise ValueError("numeric init_pop must be a positive temperature")
        initial = _temperature(init_pop, teunit)
        if initial.ndim != 0:
            raise ValueError("numeric init_pop must be a scalar temperature")
        return float(initial)

    @staticmethod
    def _broadcast_plasma_parameters(temperatures, tau, teunit):
        temp = _temperature(temperatures, teunit)
        age = _positive(tau, "ionization age", zero=True)
        if age.ndim > 1:
            raise ValueError("ionization age must be a scalar or 1D array")
        try:
            return np.broadcast_arrays(temp, age)
        except ValueError:
            raise ValueError("temperature and ionization age must be broadcast-compatible") from None

    def _nei_line_emissivity_by_line(
        self, selected, temperatures, tau, *, teunit="keV", init_pop="ionizing",
        abundances=None, abundset=None, pshock=False, tau_lower=0.,
    ):
        temp, age = self._broadcast_plasma_parameters(temperatures, tau, teunit)
        initial = self._initial_population(init_pop, teunit)
        lower = _positive(tau_lower, "lower ionization age", zero=True)
        if lower.ndim != 0:
            raise ValueError("lower ionization age must be a scalar")
        if pshock and np.any(age <= lower):
            raise ValueError("upper ionization age must exceed lower ionization age")

        values = np.zeros((len(selected), temp.size), dtype=float)
        with self._using():
            session = self._configure_session(
                self._nei(pshock=pshock), abundances, abundset=abundset
            )
            self._check_nei_temperature(session, temp)
            for index, (temperature, upper_age) in enumerate(
                zip(temp.reshape(-1), age.reshape(-1))
            ):
                for line_index, line in enumerate(selected):
                    if pshock:
                        result = session.return_line_emissivity(
                            float(temperature), float(upper_age),
                            line.element, line.ion, line.upper, line.lower,
                            tau_llist=float(lower), init_pop=initial,
                            teunit="keV", apply_aeff=False,
                        )
                    else:
                        result = session.return_line_emissivity(
                            float(temperature), float(upper_age),
                            line.element, line.ion, line.upper, line.lower,
                            init_pop=initial, teunit="keV", apply_aeff=False,
                        )
                    emissivity = np.asarray(result["epsilon"], dtype=float)
                    if emissivity.size != 1 or not np.isfinite(emissivity).all():
                        raise ValueError("PyAtomDB returned invalid NEI line emissivity")
                    values[line_index, index] = float(emissivity.reshape(-1)[0])
        return values.reshape((len(selected),) + temp.shape)

    def nei_line_emissivity(
        self, lines, temperatures, ionization_age, *, teunit="keV",
        init_pop="ionizing", abundances=None, abundset=None,
    ):
        """Return NEI line/blend emissivities for temperatures and ion ages.

        Ionization age is ``n_e * t`` in cm^-3 s. ``init_pop`` may be
        ``'ionizing'``, ``'recombining'``, or an initial equilibrium
        temperature in ``teunit``. Temperature and age arrays are broadcast
        together. Returned emissivities are photons cm^3/s.
        """
        selected = _line_ids(lines)
        if selected is None or len(set(selected)) != len(selected):
            raise ValueError("provide distinct transition identities")
        temp, age = self._broadcast_plasma_parameters(
            temperatures, ionization_age, teunit
        )
        initial = self._initial_population(init_pop, teunit)
        values = self._nei_line_emissivity_by_line(
            selected, temp, age, teunit="keV", init_pop=initial,
            abundances=abundances, abundset=abundset,
        ).sum(axis=0)
        return float(values) if values.ndim == 0 else values

    def pshock_line_emissivity(
        self, lines, temperatures, tau_upper, *, tau_lower=0., teunit="keV",
        init_pop="ionizing", abundances=None, abundset=None,
    ):
        """Return PShock line/blend emissivities integrated over ion age.

        Ionization-age bounds are ``n_e * t`` in cm^-3 s. Temperature and
        upper-age arrays are broadcast together; lower age is a scalar.
        """
        selected = _line_ids(lines)
        if selected is None or len(set(selected)) != len(selected):
            raise ValueError("provide distinct transition identities")
        temp, age = self._broadcast_plasma_parameters(
            temperatures, tau_upper, teunit
        )
        initial = self._initial_population(init_pop, teunit)
        values = self._nei_line_emissivity_by_line(
            selected, temp, age, teunit="keV", init_pop=initial,
            abundances=abundances, abundset=abundset, pshock=True,
            tau_lower=tau_lower,
        ).sum(axis=0)
        return float(values) if values.ndim == 0 else values

    def _find_nei_lines(
        self, lo, hi, *, temperature, tau, tau_lower=0., pshock=False,
        init_pop="ionizing", unit="Angstrom", teunit="keV", element=None,
        ion=None, abundances=None, abundset=None, min_emissivity=None,
        brightest=None,
    ):
        elo, ehi, _ = convert_bins(
            np.atleast_1d(lo), np.atleast_1d(hi), unit, "keV"
        )
        if (elo.size != 1 or not np.isfinite([lo, hi]).all()
                or lo <= 0 or hi <= lo):
            raise ValueError("provide finite scalar search bounds with 0 < lo < hi")
        temp = _temperature(temperature, teunit)
        if temp.ndim != 0:
            raise ValueError("line search requires a scalar temperature")
        upper_age = _positive(tau, "ionization age", zero=True)
        if upper_age.ndim != 0:
            raise ValueError("line search requires a scalar ionization age")
        lower_age = _positive(tau_lower, "lower ionization age", zero=True)
        if lower_age.ndim != 0:
            raise ValueError("lower ionization age must be a scalar")
        if pshock and upper_age <= lower_age:
            raise ValueError("upper ionization age must exceed lower ionization age")
        initial = self._initial_population(init_pop, teunit)
        elements = _elements(element)
        ions = None if ion is None else tuple(
            _integer(value, "ion", 1, 31) for value in np.atleast_1d(ion)
        )
        if min_emissivity is not None:
            min_emissivity = float(_positive(
                min_emissivity, "min_emissivity", zero=True
            ))
        if brightest is not None:
            brightest = _integer(brightest, "brightest", 1, 2**31 - 1)

        with self._using():
            session = self._configure_session(
                self._nei(pshock=pshock), abundances, abundset=abundset
            )
            self._check_nei_temperature(session, temp)
            if pshock:
                raw = session.return_linelist(
                    float(temp), float(upper_age),
                    [float(elo[0]), float(ehi[0])],
                    tau_l=float(lower_age), init_pop=initial,
                    specunit="keV", teunit="keV", apply_aeff=False,
                )
            else:
                raw = session.return_linelist(
                    float(temp), float(upper_age),
                    [float(elo[0]), float(ehi[0])],
                    init_pop=initial, specunit="keV", teunit="keV",
                    apply_aeff=False,
                )
        rows = _canonical(raw)
        mask = (
            (rows["Energy"] >= elo[0]) & (rows["Energy"] <= ehi[0])
            & np.isin(rows["Element"], elements) & (rows["Lambda"] > 0)
        )
        if ions is not None:
            mask &= np.isin(rows["Ion"], ions)
        if min_emissivity is not None:
            mask &= rows["Epsilon"] >= min_emissivity
        rows = rows[mask].copy()
        if brightest is None:
            order = np.argsort(rows["Lambda"], kind="stable")
        else:
            order = np.argsort(-rows["Epsilon"], kind="stable")[:brightest]
        return rows[order]

    def find_nei_lines(
        self, lo, hi, *, temperature, ionization_age, teunit="keV",
        init_pop="ionizing", **kwargs,
    ):
        """Find and rank lines from an explicit single-zone NEI plasma."""
        return self._find_nei_lines(
            lo, hi, temperature=temperature, tau=ionization_age,
            teunit=teunit, init_pop=init_pop, **kwargs,
        )

    def find_pshock_lines(
        self, lo, hi, *, temperature, tau_upper, tau_lower=0., teunit="keV",
        init_pop="ionizing", **kwargs,
    ):
        """Find and rank lines from an explicit plane-parallel shock plasma."""
        return self._find_nei_lines(
            lo, hi, temperature=temperature, tau=tau_upper,
            tau_lower=tau_lower, pshock=True, teunit=teunit,
            init_pop=init_pop, **kwargs,
        )

    def query_plasma_lines(self, lo, hi, *, model=None, dataset_id=None,
                           session=None, components=None, unit="Angstrom",
                           brightest=None):
        """Calculate AtomDB line contributions for the current Sherpa model.

        By default, capture the current source from ``sherpa.astro.ui``.
        Supply ``model`` for a source expression or ``dataset_id`` and
        optional ``session`` to select a session source. Alternatively,
        ``components`` accepts one or more previously captured
        ``CapturedPlasmaComponent`` snapshots; this avoids reevaluating XSPEC
        and is useful for retaining snapshots for later inspection/comparison.

        Transitions are discovered at every captured CIE temperature or
        captured NEI/PShock state. Results preserve component order.
        ``brightest`` optionally keeps the top N transitions ranked by total
        contribution across components; all discovered lines are returned by
        default.

        Captured XSPEC abundance-table names are mapped to corresponding local
        AtomDB sets before applying model abundance parameters. Unsupported
        tables or tables absent locally raise ValueError. Returned contributions
        are emissivity contributions, not absolute line fluxes. Additional
        XSPEC XSET abundance overrides are not interpreted.
        """
        from .xspec_capture import (
            CapturedPlasmaComponent, _NEI_INITIAL_POPULATIONS, _NEI_TYPES,
            _PSHOCK_TYPES,
        )

        if components is None:
            if model is not None and (dataset_id is not None or session is not None):
                raise ValueError("supply a model or a dataset/session, not both")
            from .xspec_capture import capture_plasma_model
            components = capture_plasma_model(
                model, dataset_id=dataset_id, session=session
            )
        elif model is not None or dataset_id is not None or session is not None:
            raise ValueError(
                "supply captured components or a model/dataset/session, not both"
            )
        elif isinstance(components, CapturedPlasmaComponent):
            components = (components,)
        else:
            components = tuple(components)

        if not components:
            raise ValueError("provide at least one captured plasma component")
        if any(not isinstance(component, CapturedPlasmaComponent)
               for component in components):
            raise TypeError("components must contain CapturedPlasmaComponent values")
        if brightest is not None:
            brightest = _integer(brightest, "brightest", 1, 2**31 - 1)

        settings = []
        discovered = {}
        for component in components:
            abundset = _xspec_abundance_set(component.abundance_table)
            model_type = component.model_type.lower()
            if model_type in _NEI_TYPES or model_type in _PSHOCK_TYPES:
                abundances, suppressed = _component_abundances(
                    component, allow_missing=True
                )
                parameters = {
                    str(name).lower(): float(value)
                    for name, value in component.parameters.items()
                }
                temperature = parameters.get("kt")
                if temperature is None or temperature <= 0:
                    raise ValueError(
                        f"captured {model_type} component requires positive kT"
                    )
                initial = _NEI_INITIAL_POPULATIONS.get(model_type, "ionizing")
                if initial == "initial_temperature":
                    initial = parameters.get("kt_init")
                    if initial is None or initial <= 0:
                        raise ValueError(
                            f"captured {model_type} component requires positive kT_init"
                        )
                if model_type in _PSHOCK_TYPES:
                    lower, upper = parameters.get("tau_l"), parameters.get("tau_u")
                    if lower is None or upper is None or lower < 0 or upper <= lower:
                        raise ValueError(
                            f"captured {model_type} component requires 0 <= Tau_l < Tau_u"
                        )
                    rows = self.find_pshock_lines(
                        lo, hi, temperature=temperature, tau_upper=upper,
                        tau_lower=lower, unit=unit, init_pop="ionizing",
                        abundances=abundances, abundset=abundset,
                    )
                else:
                    tau = parameters.get("tau")
                    if tau is None or tau < 0:
                        raise ValueError(
                            f"captured {model_type} component requires nonnegative Tau"
                        )
                    rows = self.find_nei_lines(
                        lo, hi, temperature=temperature,
                        ionization_age=tau, unit=unit, init_pop=initial,
                        abundances=abundances, abundset=abundset,
                    )
                settings.append((abundset, abundances, "nei", initial))
                for row in rows:
                    if int(row["Element"]) in suppressed:
                        continue
                    line = LineID.from_row(row)
                    if line not in discovered:
                        copy = row.copy()
                        copy["Epsilon"] = np.nan
                        discovered[line] = copy
            elif component.temperatures is not None:
                abundances, suppressed = _component_abundances(component)
                settings.append((abundset, abundances, "cie", None))
                for temperature in component.temperatures:
                    rows = self.find_lines(
                        lo, hi, unit=unit, kT=float(temperature),
                        abundances=abundances, abundset=abundset,
                    )
                    for row in rows:
                        if int(row["Element"]) in suppressed:
                            continue
                        line = LineID.from_row(row)
                        if line not in discovered:
                            copy = row.copy()
                            copy["Epsilon"] = np.nan
                            discovered[line] = copy
            else:
                raise ValueError(
                    f"captured model {model_type!r} has no supported AtomDB "
                    "emissivity mapping"
                )

        line_ids = sorted(
            discovered,
            key=lambda line: (discovered[line]["Lambda"], line.element, line.ion,
                              line.upper, line.lower),
        )
        lines = (np.array([discovered[line] for line in line_ids], dtype=_LINE_DTYPE)
                 if line_ids else np.empty(0, dtype=_LINE_DTYPE))

        component_results = []
        for component, (abundset, abundances, kind, initial) in zip(components, settings):
            if line_ids and kind == "cie":
                emissivities = self._line_emissivity_by_line(
                    line_ids, component.temperatures,
                    abundances=abundances, abundset=abundset,
                )
                values = emissivities @ component.weights
            elif line_ids and kind == "nei":
                parameters = {
                    str(name).lower(): float(value)
                    for name, value in component.parameters.items()
                }
                temperature = parameters["kt"]
                if component.model_type.lower() in _PSHOCK_TYPES:
                    emissivities = self._nei_line_emissivity_by_line(
                        line_ids, temperature, parameters["tau_u"],
                        init_pop=initial, abundances=abundances,
                        abundset=abundset, pshock=True,
                        tau_lower=parameters["tau_l"],
                    )
                else:
                    emissivities = self._nei_line_emissivity_by_line(
                        line_ids, temperature, parameters["tau"],
                        init_pop=initial, abundances=abundances,
                        abundset=abundset,
                    )
                values = emissivities.reshape(len(line_ids))
            else:
                values = np.empty(0, dtype=float)
            component_results.append(ComponentLineContributions(
                component.name, component.model_type, values
            ))

        total = (np.sum([item.values for item in component_results], axis=0)
                 if line_ids else np.empty(0, dtype=float))
        lines["ModelContribution"] = total
        if brightest is not None and len(lines) > brightest:
            order = np.argsort(-total, kind="stable")[:brightest]
            lines = lines[order]
            component_results = [
                ComponentLineContributions(
                    item.name, item.model_type, item.values[order]
                )
                for item in component_results
            ]
            total = total[order]
        return PlasmaLineContributions(lines, tuple(component_results), total)

    def ratio_emissivity(self, numerator, denominator, temperatures, **kwargs):
        """Line/blend emissivity ratio; zero denominators return NaN."""
        num = np.asarray(self.line_emissivity(numerator, temperatures, **kwargs))
        den = np.asarray(self.line_emissivity(denominator, temperatures, **kwargs))
        result = np.full(num.shape, np.nan)
        np.divide(num, den, out=result, where=den != 0)
        return float(result) if result.ndim == 0 else result


    def read_model(self, model=None, *, dataset_id=None, session=None):
        """Read current built-in APEC parameters without evaluating the model.

        With no model, read the source of dataset_id in the supplied Sherpa
        session (or the global UI). Returns detached PlasmaState snapshots.
        No database files or PyAtomDB session are needed for this operation.
        """
        from .plasma_state import read_plasma_model
        return read_plasma_model(model, dataset_id=dataset_id, session=session)

    def close(self):
        """Close opened FITS files and remove generated cache files."""
        with _LOCK:
            seen = set()
            def close_tree(value):
                if id(value) in seen:
                    return
                seen.add(id(value))
                if isinstance(value, dict):
                    for child in value.values():
                        close_tree(child)
                elif hasattr(value, "close"):
                    value.close()
            close_tree(self._datacache)
            sessions = ([self._session] if self._session is not None else [])
            sessions.extend(self._nei_sessions.values())
            for session in sessions:
                for value in (session.linedata, session.cocodata,
                              session.datacache, session.spectra.datacache):
                    close_tree(value)
            close_tree(self._sherpa_eigen_hdul)
            self._session = None
            self._nei_sessions.clear()
            self._sherpa_eigen_hdul = None
            self._sherpa_eigen_path = None
            if self._scratch is not None:
                self._scratch.cleanup()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
