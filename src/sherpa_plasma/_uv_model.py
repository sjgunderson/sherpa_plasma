"""Sherpa composite for AtomDB-backed He-like triplet pumping."""

import hashlib
import json
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from sherpa.astro import xspec
from sherpa.models.model import ArithmeticModel, CompositeModel
from sherpa.models.parameter import Parameter

from ._uv_coverage import _validated_dem_weights
from ._uv_lines import (
    _atomic_line_wavelengths, _integrate_triplet, _prepare_triplet_grid,
    _read_native_lines,
)
from ._uv_projection import _unbroadened_line_bins
from ._uv_pumping import _transfer_triplet_photons
from .plasma import AtomDB, LineID, _element, _SYMBOLS
from .xspec_capture import _capture_cie_evaluation, _CAPTURE_LOCK


_RELEVANT_XSETS = frozenset({
    "APECROOT", "APECLOGINTERP", "APECTHERMAL", "APECMULTITHREAD", "APECVELOCITY",
    "APECMINFLUX", "APECBROADPSEUDO", "APECNOLINES", "APECUSENEI",
    "APECEEBREMSS", "APECDOECOR", "APECUSESPEX", "APEC_TRACE_ABUND",
    "LINECRITLEVEL",
})
_MANIFESTS = ("_uv_atomdb_3_0_9.json", "_uv_atomdb_3_1_3.json")
_MAIN_ELEMENTS = frozenset({"C", "N", "O", "Ne", "Mg", "Al", "Si", "S",
                            "Ar", "Ca", "Fe", "Ni"})


@lru_cache(maxsize=16)
def _file_sha256(path, _identity):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state():
    settings = xspec.get_xsxset()
    table = xspec.get_xsabund()
    # XSPEC calls every custom vector "file". A reload can change any element
    # without changing that name, including elements affecting cooling weights.
    # Named tables are immutable in XSPEC; keep their fitting path inexpensive.
    custom = None
    if table == "file":
        custom = tuple(sorted((symbol, float(value))
                              for symbol, value in xspec.get_xsabundances().items()))
        if any(not np.isfinite(value) or value < 0 for _, value in custom):
            raise ValueError("XSPEC abundance values must be finite and nonnegative")
    return (xspec.get_xsversion(), table, custom,
            tuple(sorted((key, value) for key, value in settings.items()
                         if key.startswith("APEC") or key in _RELEVANT_XSETS)))


def _abundance_scales(elements):
    """Convert reference APEC emissivities to the active XSPEC composition."""
    reference = xspec.get_xsabundances("angr")
    actual = xspec.get_xsabundances()
    scales = {}
    for z in elements:
        symbol = _SYMBOLS[z]
        value, baseline = float(actual[symbol]), float(reference[symbol])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{symbol} XSPEC abundance must be finite and nonnegative")
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"{symbol} has no positive reference XSPEC abundance")
        scale = value / baseline
        if not np.isfinite(scale):
            raise ValueError(f"{symbol} XSPEC abundance scaling is not finite")
        scales[z] = scale
    return scales


def _file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _root():
    setting = xspec.get_xsxset().get("APECROOT")
    if not setting or not Path(setting).is_absolute():
        raise ValueError("set APECROOT to an absolute APEC file root")
    return Path(setting).resolve()


@lru_cache(maxsize=16)
def _load_grid(line_file, lines, file_hash, fallback_wavelengths=()):
    return _prepare_triplet_grid(
        _read_native_lines(line_file, lines), fallback_wavelengths)


@lru_cache(maxsize=4)
def _load_broad_profiles(library_path, root, element, ground, upper, minimum, maximum,
                         file_hash, continuum_hash, identities, wavelengths=()):
    from . import _xspec_profiles

    paths = (Path(root + "_line.fits"), Path(root + "_coco.fits"))
    if tuple(_file_identity(path) for path in paths) != identities:
        raise RuntimeError("APEC files changed before native profile loading")
    profiles = _xspec_profiles.create(
        library_path, root, element, ground, upper, minimum, maximum, wavelengths)
    if tuple(_file_identity(path) for path in paths) != identities:
        raise RuntimeError("APEC files changed during native profile loading")
    return profiles


def _temperature_interval(grid):
    return float(grid.temperatures[0]), float(grid.temperatures[-1])


def _ratio_diagnostic(forbidden, intercombination, pumping):
    if intercombination > 0:
        r0 = float(forbidden / intercombination)
        return r0, r0 / (1 + pumping), "defined"
    if forbidden > 0:
        return float("inf"), float("inf"), "no_intercombination_emission"
    return None, None, "no_triplet_emission"


def _installed_manifest(provider, selected, line_file, hashes):
    """Build a calibration from the same APEC files and local LV/LA data."""
    db = provider.db
    for path, expected in zip((db.linefile, db.cocofile), hashes):
        if _file_sha256(path, _file_identity(path)) != expected:
            raise ValueError("AtomDB provider line and continuum files must match active APEC files")
    filemap = db.filemap
    files = [filemap]
    identities = [_file_identity(filemap)]
    manifest = {
        "version": db.info["version"],
        "line_file_sha256": hashes[0],
        "continuum_file_sha256": hashes[1],
        "filemap_sha256": _file_sha256(filemap, identities[0]),
        "ions": {},
    }
    for z in selected:
        level, radiative = provider.atomic_files(z)
        atomic_files = (level, radiative)
        atomic_identities = tuple(_file_identity(path) for path in atomic_files)
        calibration = provider._calibration(z, atomic_files)
        upper = (calibration.metastable, *calibration.triplet_p[1:])
        lines = tuple(LineID(z, z - 1, index, calibration.ground) for index in upper)
        wavelengths = _atomic_line_wavelengths(level, radiative, lines)
        grid = _load_grid(str(line_file), lines, hashes[0], wavelengths)
        low, high = _temperature_interval(grid)
        manifest["ions"][str(z)] = {
            "calibration": asdict(calibration),
            "level_file": str(level),
            "level_file_sha256": _file_sha256(level, atomic_identities[0]),
            "radiative_file": str(radiative),
            "radiative_file_sha256": _file_sha256(radiative, atomic_identities[1]),
            "temperature_min_keV": low,
            "temperature_max_keV": high,
            "line_wavelengths_A": tuple(grid.wavelengths[0]),
        }
        files.extend(atomic_files)
        identities.extend(atomic_identities)
    if tuple(_file_identity(path) for path in files) != tuple(identities):
        raise RuntimeError("AtomDB atomic files changed during triplet calibration")
    return manifest, tuple(files), tuple(identities)


class _PumpingParameters(ArithmeticModel):
    def __init__(self, name, elements):
        parameters = []
        for z in elements:
            p = Parameter(name, "p_" + _SYMBOLS[z], 0., min=0., hard_min=0.)
            setattr(self, p.name, p)
            parameters.append(p)
        super().__init__(name, parameters)


class _TripletPumpingModel(CompositeModel, ArithmeticModel):
    """Source-level He-like triplet pumping correction for a plasma component."""

    def __init__(self, child, elements, *, name=None, atomdb=None,
                 coverage_tolerance=None):
        # Accepted for compatibility with the former bounded-tail implementation.
        if coverage_tolerance is not None:
            coverage_tolerance = float(coverage_tolerance)
            if not np.isfinite(coverage_tolerance) or not 0 <= coverage_tolerance < 1:
                raise ValueError("coverage_tolerance must be finite and in [0, 1)")
        kind = type(child)
        apec_types = tuple(
            cls for cls in (getattr(xspec, name, None)
                            for name in ("XSapec", "XSvapec", "XSvvapec",
                                         "XSbapec", "XSbvapec", "XSbvvapec"))
            if cls is not None
        )
        scalar_apec = (xspec.XSapec, getattr(xspec, "XSbapec", None))
        full_apec = (xspec.XSvvapec, getattr(xspec, "XSbvvapec", None))
        wdem_types = tuple(
            cls for cls in (getattr(xspec, name, None)
                            for name in ("XSwdem", "XSvwdem", "XSvvwdem",
                                         "XSbwdem", "XSbvwdem", "XSbvvwdem"))
            if cls is not None
        )
        broad_types = tuple(
            cls for cls in (getattr(xspec, name, None)
                            for name in ("XSbapec", "XSbvapec", "XSbvvapec",
                                         "XSbwdem", "XSbvwdem", "XSbvvwdem"))
            if cls is not None
        )
        thermal_default = kind in broad_types
        is_wdem = kind in wdem_types
        is_mkcflow = kind is getattr(xspec, "XSmkcflow", None)
        is_dem = is_wdem or is_mkcflow
        if kind not in apec_types and not is_dem:
            raise TypeError("triplet pumping requires a supported XSPEC APEC, wdem, or XSmkcflow component")
        if name is None:
            name = "uv_" + child.name
        if xspec.get_xsversion() != "12.14.0k":
            raise ValueError("triplet pumping is only validated with XSPEC 12.14.0k")
        if isinstance(elements, (str, int)):
            elements = (elements,)
        selected = tuple(_element(value) for value in elements)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("select at least one distinct element")
        settings = xspec.get_xsxset()
        # Let XSPEC resolve the thermal switch itself, including explicit "no".
        # Even non-b models need native profiles when this override is present.
        is_broadened = thermal_default or "APECTHERMAL" in settings
        allowed = {"APECROOT", "APECTHERMAL", "APECMULTITHREAD"}
        if kind not in full_apec and not is_dem:
            allowed.add("APEC_TRACE_ABUND")
        unsupported = sorted(key for key in settings
                             if key not in allowed
                             and (key.startswith("APEC") or key in _RELEVANT_XSETS))
        if unsupported:
            raise ValueError("unsupported XSPEC settings: " + ", ".join(unsupported))
        trace = settings.get("APEC_TRACE_ABUND")
        if trace is None:
            trace_source = ("constant", 1.)
        elif trace == "Fe":
            trace_source = ("parameter", "abundanc" if kind in scalar_apec else "fe")
        else:
            try:
                trace_value = float(trace)
            except ValueError as exc:
                raise ValueError("APEC_TRACE_ABUND must be a nonnegative number or Fe") from exc
            if not np.isfinite(trace_value) or trace_value < 0:
                raise ValueError("APEC_TRACE_ABUND must be a nonnegative number or Fe")
            trace_source = ("constant", trace_value)
        root = _root()
        line_file = Path(str(root) + "_line.fits")
        continuum_file = Path(str(root) + "_coco.fits")
        identities = (_file_identity(line_file), _file_identity(continuum_file))
        file_hash = _file_sha256(line_file, identities[0])
        continuum_hash = _file_sha256(continuum_file, identities[1])
        manifests = []
        for filename in _MANIFESTS:
            with (Path(__file__).parent / filename).open(encoding="utf-8") as file:
                manifests.append(json.load(file))
        manifest = next((item for item in manifests
                         if item["line_file_sha256"] == file_hash
                         and item["continuum_file_sha256"] == continuum_hash), None)
        atomic_files, atomic_identities = (), ()
        if atomdb is not None or manifest is None:
            from ._atomdb_pumping import AtomDBTripletProvider

            if atomdb is None:
                if not os.environ.get("ATOMDB"):
                    raise ValueError(
                        "active APEC files do not match a bundled calibration; "
                        "provide a matching AtomDB connection or set ATOMDB"
                    )
                atomdb = AtomDB()
            provider = atomdb if isinstance(atomdb, AtomDBTripletProvider) else AtomDBTripletProvider(atomdb)
            manifest, atomic_files, atomic_identities = _installed_manifest(
                provider, selected, line_file, (file_hash, continuum_hash))
        if is_dem or is_broadened:
            try:
                from sherpa.astro.xspec import _xspec
            except ImportError as exc:
                raise ImportError("native triplet projection requires Sherpa/XSPEC") from exc
            self._library_path = _xspec.__file__
        if is_dem:
            try:
                from . import _xspec_bridge
            except ImportError as exc:
                raise ImportError("DEM triplet pumping requires the optional XSPEC bridge") from exc
        if is_broadened:
            try:
                from . import _xspec_profiles
            except ImportError as exc:
                raise ImportError("thermal/broadened pumping requires the optional XSPEC profile bridge") from exc
            if (not hasattr(_xspec_profiles, "project_transfer")
                    or getattr(_xspec_profiles, "_profile_options_version", 0) < 1):
                raise ImportError(
                    "thermal/broadened pumping requires a rebuilt XSPEC profile bridge; "
                    "install with SHERPA_PLASMA_BUILD_XSPEC_BRIDGE=1")
        self._is_wdem = is_wdem
        self._is_broadened = is_broadened
        self._thermal_default = thermal_default
        self._is_mkcflow = is_mkcflow
        self._is_dem = is_dem
        self._cosmology_at_init = tuple(xspec.get_xscosmo()) if is_mkcflow else None
        self._selected = selected
        self._child = child
        self._pumping = _PumpingParameters(name, selected)
        self._state_at_init = _state()
        self._manifest = manifest
        self._entries = []
        self._abundance_sources = []
        child_parameter_names = {par.name.lower() for par in child.pars}
        abundance_scales = _abundance_scales(selected)
        for z in selected:
            record = manifest["ions"].get(str(z))
            if record is None:
                raise ValueError(f"{_SYMBOLS[z]} has no triplet atomic calibration; provide matching AtomDB data")
            calibration = record["calibration"]
            upper = (calibration["metastable"], *calibration["triplet_p"][1:])
            lines = tuple(LineID(z, z - 1, index, calibration["ground"])
                          for index in upper)
            grid = _load_grid(str(line_file), lines, file_hash,
                              tuple(record.get("line_wavelengths_A", ())))
            self._entries.append((record, grid, abundance_scales[z]))
            symbol = _SYMBOLS[z]
            if is_wdem and kind not in (getattr(xspec, "XSwdem", None),
                                        getattr(xspec, "XSbwdem", None)):
                source = (("parameter", symbol.lower())
                          if symbol.lower() in child_parameter_names
                          else ("constant", 1.))
            elif is_dem and symbol not in _MAIN_ELEMENTS and symbol != "Na":
                source = ("constant", 1.)
            elif is_dem:
                source = ("parameter", "abundanc")
            elif kind in full_apec:
                source = ("parameter", symbol.lower())
            elif symbol not in _MAIN_ELEMENTS:
                source = trace_source
            elif kind in scalar_apec:
                source = ("parameter", "abundanc")
            else:
                source = ("parameter", symbol.lower())
            self._abundance_sources.append(source)
            setattr(self, "p_" + _SYMBOLS[z], getattr(self._pumping, "p_" + _SYMBOLS[z]))
        if identities != (_file_identity(line_file), _file_identity(continuum_file)):
            raise RuntimeError("APEC data changed during UV wrapper construction")
        self._data_files = (line_file, continuum_file) + atomic_files
        self._data_identities = identities + atomic_identities
        self._child_count = len(child.pars)
        self._indices = {param.name.lower(): index for index, param in enumerate(child.pars)}
        self._child_last = None
        self._child_last_key = None
        self._child_temperatures = None
        self._child_weights = None
        self._profile_cache = {}
        # Sherpa's child cache is keyed by parameters and energy bins, not the
        # XSPEC abundance table. A reconstructed wrapper needs a fresh spectrum.
        child.cache_clear()
        CompositeModel.__init__(self, name, (child, self._pumping))

    def cache_clear(self):
        self._child_last = None
        self._child_last_key = None
        self._child_temperatures = None
        self._child_weights = None
        self._profile_cache.clear()
        super().cache_clear()

    def startup(self, cache=False):
        self._child_last = None
        self._child_last_key = None
        self._child_temperatures = None
        self._child_weights = None
        self._profile_cache.clear()
        self._child.startup(cache)
        self._pumping.startup(cache)
        super().startup(cache)

    def teardown(self):
        self._child_last = None
        self._child_last_key = None
        self._child_temperatures = None
        self._child_weights = None
        self._profile_cache.clear()
        self._child.teardown()
        self._pumping.teardown()
        super().teardown()

    def _check_state(self):
        if _state() != self._state_at_init:
            raise RuntimeError("XSPEC settings or abundances changed; reconstruct the UV wrapper")
        if tuple(_file_identity(path) for path in self._data_files) != self._data_identities:
            raise RuntimeError("APEC data files changed; restart XSPEC and reconstruct the UV wrapper")
        if self._is_mkcflow and tuple(xspec.get_xscosmo()) != self._cosmology_at_init:
            raise RuntimeError("XSPEC cosmology changed; reconstruct the UV wrapper")

    def temperature_coverage(self):
        self._check_state()
        intervals = {
            _SYMBOLS[z]: _temperature_interval(grid)
            for z, (_, grid, _) in zip(self._selected, self._entries)
        }
        active = [intervals[_SYMBOLS[z]] for z, parameter in
                  zip(self._selected, self._pumping.pars) if parameter.val > 0]
        intersection = None
        if active:
            low, high = max(interval[0] for interval in active), min(interval[1] for interval in active)
            intersection = (low, high) if low <= high else ()
        return {"ions_keV": intervals, "active_intersection_keV": intersection,
                "policy": "full_dem", "outside_native_grid": "nearest_endpoint"}

    def diagnostics(self):
        self._check_state()
        if self._is_dem:
            return self._dem_diagnostics()
        temperature = float(self._child.pars[self._indices["kt"]].val)
        results = []
        for z, parameter, (record, grid, abundance_scale) in zip(
                self._selected, self._pumping.pars, self._entries):
            emissivities, _ = _integrate_triplet(grid, [temperature], [1.])
            f, i1, i2 = emissivities * abundance_scale
            r0, ratio, status = _ratio_diagnostic(f, i1 + i2, float(parameter.val))
            c = record["calibration"]
            results.append({
                "element": _SYMBOLS[z],
                "ion": z - 1,
                "p": float(parameter.val),
                "R0": r0,
                "R": ratio,
                "ratio_status": status,
                "intrinsic_triplet_fluxes": (float(f), float(i1), float(i2)),
                "abundance_scale_from_angr": float(abundance_scale),
                "atomdb_version": self._manifest["version"],
                "line_file_sha256": self._manifest["line_file_sha256"],
                "continuum_file_sha256": self._manifest["continuum_file_sha256"],
                "filemap_sha256": self._manifest["filemap_sha256"],
                "level_file": record["level_file"],
                "level_file_sha256": record["level_file_sha256"],
                "radiative_file": record["radiative_file"],
                "radiative_file_sha256": record["radiative_file_sha256"],
                "forbidden_rate_s-1": c["forbidden_rate"],
                "effective_branching": c["effective_branching"],
                "intercombination_redistribution": tuple(c["redistribution"]),
                "temperature_range_keV": _temperature_interval(grid),
                "omitted_decay_fraction": tuple(c["omitted_decay_fraction"]),
                "profile": ("native XSPEC 12.14.0k projection" if self._is_broadened
                            else "unbroadened XSPEC 12.14.0k"),
            })
        return tuple(results)

    def _dem_diagnostics(self):
        from . import _xspec_bridge

        trial = np.array([par.val for par in self._child.pars], dtype=float)
        self._validate_dem_trial(trial)
        edges = np.geomspace(.01, 100., 1001)
        _, temperatures, weights = _capture_cie_evaluation(
            self._child, type(self._child).__name__.lower(), trial,
            edges[:-1], edges[1:], _xspec_bridge, self._library_path,
        )
        temperatures, weights = _validated_dem_weights(temperatures, weights)
        results = []
        for z, parameter, (record, grid, abundance_scale) in zip(
                self._selected, self._pumping.pars, self._entries):
            active = weights > 0
            emissivities, _ = _integrate_triplet(grid, temperatures, weights)
            f, i1, i2 = emissivities * abundance_scale
            r0, ratio, status = _ratio_diagnostic(f, i1 + i2, float(parameter.val))
            results.append({
                "element": _SYMBOLS[z],
                "ion": z - 1,
                "p": float(parameter.val),
                "R0_integrated": r0,
                "R_integrated": ratio,
                "ratio_status": status,
                "intrinsic_triplet_fluxes": (float(f), float(i1), float(i2)),
                "abundance_scale_from_angr": float(abundance_scale),
                "temperature_range_keV": (float(temperatures[active].min()),
                                          float(temperatures[active].max())),
                "coverage_keV": _temperature_interval(grid),
                "coverage_policy": "full_dem",
                "temperature_count": int(np.count_nonzero(active)),
                "atomdb_version": self._manifest["version"],
                "line_file_sha256": self._manifest["line_file_sha256"],
                "continuum_file_sha256": self._manifest["continuum_file_sha256"],
                "profile": ("native XSPEC 12.14.0k projection" if self._is_broadened
                            else "unbroadened XSPEC 12.14.0k"),
            })
        return tuple(results)

    def _validate_dem_trial(self, trial):
        if trial[self._indices["switch"]] != 2:
            raise ValueError("DEM triplet pumping requires switch=2 (AtomDB)")
        if self._is_wdem and trial[self._indices["nh"]] != 1:
            raise ValueError("wdem triplet pumping requires nH=1")
        if self._is_mkcflow and trial[self._indices["redshift"]] <= 0:
            raise ValueError("XSmkcflow triplet pumping requires positive redshift")

    def calc(self, p, lo, hi, **kwargs):
        self._check_state()
        if kwargs:
            raise ValueError("triplet-pumping wrapper does not support additional evaluation keywords")
        p = np.asarray(p, dtype=float)
        if p.ndim != 1 or len(p) != len(self.pars) or not np.isfinite(p).all():
            raise ValueError("invalid trial parameter vector")
        pumping = p[self._child_count:]
        if np.any(pumping < 0):
            raise ValueError("pumping parameters must be nonnegative")
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        child_parameters = p[:self._child_count]
        if self._is_dem and np.any(pumping):
            self._validate_dem_trial(child_parameters)
        child_key = (child_parameters.tobytes(), lo.shape, lo.tobytes(), hi.shape, hi.tobytes())
        if child_key != self._child_last_key:
            self._profile_cache.clear()
            if self._is_dem and np.any(pumping):
                from . import _xspec_bridge
                flux, temperatures, weights = _capture_cie_evaluation(
                    self._child, type(self._child).__name__.lower(), child_parameters, lo, hi,
                    _xspec_bridge, self._library_path,
                )
                self._child_temperatures = np.asarray(temperatures, dtype=float)
                self._child_weights = np.asarray(weights, dtype=float)
            else:
                flux = self._child.calc(child_parameters, lo, hi)
                self._child_temperatures = self._child_weights = None
            self._child_last = np.array(flux, copy=True)
            self._child_last_key = child_key
        elif self._is_dem and np.any(pumping) and self._child_temperatures is None:
            from . import _xspec_bridge
            flux, temperatures, weights = _capture_cie_evaluation(
                self._child, type(self._child).__name__.lower(), child_parameters, lo, hi,
                _xspec_bridge, self._library_path,
            )
            self._child_last = np.array(flux, copy=True)
            self._child_temperatures = np.asarray(temperatures, dtype=float)
            self._child_weights = np.asarray(weights, dtype=float)
        base = self._child_last
        if not np.any(pumping):
            return base.copy()
        names = self._indices
        if self._is_dem:
            temperatures, weights = _validated_dem_weights(
                self._child_temperatures, self._child_weights,
            )
        else:
            temperatures = np.array([child_parameters[names["kt"]]], dtype=float)
            weights = np.ones(1)
        norm = child_parameters[names["norm"]]
        redshift = child_parameters[names["redshift"]]
        if norm < 0 or redshift <= -1:
            raise ValueError("norm must be nonnegative and redshift greater than -1")
        output = base.copy()
        for z, value, (record, grid, abundance_scale), source in zip(
                self._selected, pumping, self._entries, self._abundance_sources):
            if value == 0:
                continue
            abundance = (child_parameters[names[source[1]]]
                         if source[0] == "parameter" else source[1])
            if abundance < 0:
                raise ValueError("element abundance must be nonnegative")
            active = weights > 0
            if abundance_scale == 0 or abundance == 0 or norm == 0:
                continue
            calibration = record["calibration"]
            eta1, eta2 = calibration["redistribution"]
            (f, i1, i2), wavelengths = _integrate_triplet(grid, temperatures, weights)
            transfer = _transfer_triplet_photons(f, i1, i2, value)
            if transfer == 0:
                continue
            if self._is_broadened:
                if (lo.ndim != 1 or hi.ndim != 1 or len(lo) < 2 or len(lo) != len(hi)
                        or not np.isfinite(lo).all() or not np.isfinite(hi).all()
                        or np.any(lo <= 0) or np.any(hi <= lo)
                        or not np.array_equal(hi[:-1], lo[1:])):
                    raise ValueError("XSPEC projection requires at least two contiguous ordered energy bins")
                from . import _xspec_profiles

                upper = (calibration["metastable"], *calibration["triplet_p"][1:])
                velocity = (float(child_parameters[names["velocity"]])
                            if "velocity" in names else 0.)
                projected = self._profile_cache.get(z)
                if projected is None:
                    with _CAPTURE_LOCK:
                        profiles = _load_broad_profiles(
                            self._library_path, str(_root()), z, calibration["ground"],
                            upper, *_temperature_interval(grid),
                            self._manifest["line_file_sha256"],
                            self._manifest["continuum_file_sha256"], self._data_identities[:2],
                            tuple(wavelengths))
                        projected = np.asarray(_xspec_profiles.project(
                            profiles, np.concatenate((lo[:1], hi)), temperatures[active], weights[active],
                            redshift, velocity, self._thermal_default))
                    self._check_state()
                    self._profile_cache[z] = projected
                additions = []
                donor = None
                for index, flux in ((1, i1), (2, i2)):
                    if flux > 0:
                        additions.append(projected[index] / flux)
                    else:
                        if donor is None:
                            donor = self._profile_cache.get((z, "donor"))
                            if donor is None:
                                with _CAPTURE_LOCK:
                                    profiles = _load_broad_profiles(
                                        self._library_path, str(_root()), z, calibration["ground"],
                                        upper, *_temperature_interval(grid),
                                        self._manifest["line_file_sha256"],
                                        self._manifest["continuum_file_sha256"],
                                        self._data_identities[:2], tuple(wavelengths))
                                    donor = np.asarray(_xspec_profiles.project_transfer(
                                        profiles, np.concatenate((lo[:1], hi)),
                                        temperatures[active], weights[active], redshift,
                                        velocity, self._thermal_default))
                                self._check_state()
                                self._profile_cache[(z, "donor")] = donor
                        additions.append(donor[index] / f)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    # Aped::SumEqSpectra already converts these native profiles
                    # from the reference abundances to the active XSPEC table.
                    output += abundance * norm * transfer * (
                        -projected[0] / f + eta1 * additions[0] + eta2 * additions[1])
                continue
            bins = _unbroadened_line_bins(lo, hi, wavelengths, redshift)
            transfer *= 1.e14 * abundance_scale * abundance * norm / (1 + redshift)
            for index, coefficient in zip(bins, (-1., eta1, eta2)):
                if index >= 0:
                    output[index] += coefficient * transfer
        return output


def wrap_triplet_pumping(child, elements, *, name=None, atomdb=None,
                         coverage_tolerance=None):
    """Wrap a supported component with AtomDB-backed triplet suppression.

    ``atomdb`` may be an AtomDB connection or AtomDBTripletProvider whose
    line and continuum files match the active XSPEC APEC files. Without one,
    a bundled calibration is used when available; otherwise $ATOMDB is used.

    Each transition is integrated independently over the full captured DEM.
    Missing native line entries contribute zero. Temperatures beyond the native
    table use its nearest endpoint, matching XSPEC. ``coverage_tolerance`` is
    accepted for compatibility and has no effect; no DEM bins are discarded.

    Any active XSPEC abundance table or custom vector is supported. Choose it
    before constructing the wrapper; changing it requires a new wrapper.
    APECTHERMAL and APECMULTITHREAD are honored by the child and native line
    projector. Set these before construction; changing them requires a new
    wrapper. Native profiles are required for thermal and b-family models.
    """
    return _TripletPumpingModel(
        child, elements, name=name, atomdb=atomdb,
        coverage_tolerance=coverage_tolerance)

