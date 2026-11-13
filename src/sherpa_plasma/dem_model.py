"""AtomDB-backed Sherpa models for native, user-defined CIE DEMs."""
from collections import OrderedDict
from collections.abc import Mapping
import ctypes
import math
from pathlib import Path
import shutil
import sys
import tempfile
from threading import RLock

import numpy as np

from .cie_model import (
    _ERROR_BUFFER_SIZE,
    _RESERVED_PARAMETER_NAMES,
    _SHERPA_IMPORT_ERROR,
    _Parameter,
    _compile_shared_library,
    _float64_vector,
)
from .plasma import AtomDB, _positive
from ._abundances import _resolve_abundance_spec

if _SHERPA_IMPORT_ERROR is None:
    from sherpa.models.model import ArithmeticModel
else:
    class ArithmeticModel:
        pass


_KEV_TO_K = 1.160451812e7
_ENGINE_LOCK = RLock()
_ENGINE = None
_ENGINE_DIRECTORY = None


def _native_engine():
    global _ENGINE, _ENGINE_DIRECTORY
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        source = Path(__file__).with_name("dem_cie_engine.cpp")
        _ENGINE_DIRECTORY = tempfile.TemporaryDirectory(
            prefix="sherpa-plasma-dem-engine-"
        )
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        output = Path(_ENGINE_DIRECTORY.name) / ("dem_cie_engine" + suffix)
        try:
            library_path = _compile_shared_library(source, output)
            library = ctypes.CDLL(str(library_path))
            function = library.sherpa_plasma_dem_cie_model_v1
        except (OSError, RuntimeError, AttributeError):
            _ENGINE_DIRECTORY.cleanup()
            _ENGINE_DIRECTORY = None
            raise

        double_pointer = ctypes.POINTER(ctypes.c_double)
        dem_function_pointer = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_size_t,
            double_pointer,
            double_pointer,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        )
        function.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            double_pointer,
            double_pointer,
            ctypes.c_size_t,
            double_pointer,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_double,
            dem_function_pointer,
            double_pointer,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        _ENGINE = (library, function, dem_function_pointer)
        return _ENGINE


def compile_dem_function(
    source,
    output,
    *,
    compiler=None,
    include_dirs=(),
    extra_compile_args=(),
    extra_link_args=(),
):
    """Compile a C/C++ DEM shape function using ``dem_model_abi.h``."""
    return _compile_shared_library(
        source,
        output,
        compiler=compiler,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


class _NativeDEM:
    def __init__(
        self,
        *,
        source=None,
        library=None,
        compiler=None,
        include_dirs=(),
        extra_compile_args=(),
        extra_link_args=(),
    ):
        if (source is None) == (library is None):
            raise ValueError("provide exactly one of dem_source or dem_library")
        self._directory = None
        if source is not None:
            self._directory = tempfile.TemporaryDirectory(
                prefix="sherpa-plasma-dem-"
            )
            suffix = ".dylib" if sys.platform == "darwin" else ".so"
            output = Path(self._directory.name) / ("dem_function" + suffix)
            try:
                library = _compile_shared_library(
                    source,
                    output,
                    compiler=compiler,
                    include_dirs=include_dirs,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                self._directory.cleanup()
                self._directory = None
                raise
        else:
            if (compiler is not None or include_dirs or extra_compile_args or
                    extra_link_args):
                raise ValueError("compile options are only valid with dem_source")

        path = Path(library).expanduser().resolve()
        if not path.is_file():
            if self._directory is not None:
                self._directory.cleanup()
            raise FileNotFoundError(f"compiled DEM library does not exist: {path}")
        try:
            self._library = ctypes.CDLL(str(path))
            self._function = self._library.sherpa_plasma_dem_shape_v1
        except (OSError, AttributeError) as exc:
            if self._directory is not None:
                self._directory.cleanup()
            raise RuntimeError(
                f"could not load native DEM function from {path}: {exc}"
            ) from exc

        double_pointer = ctypes.POINTER(ctypes.c_double)
        self._function.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_size_t,
            double_pointer,
            double_pointer,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        self._function.restype = ctypes.c_int


class _NativeIntegrator:
    def __init__(self, dem, *, parameters, normalization, log_tmin, log_tmax,
                 quadrature_points, mineps_offset):
        library, function, dem_function_pointer = _native_engine()
        self._library = library
        self._function = function
        self._dem_function = dem_function_pointer(dem._function)
        self._parameters = _float64_vector(parameters, "DEM parameters")
        self._normalization = float(normalization)
        self._minimum_temperature_K = 10.0 ** float(log_tmin)
        self._maximum_temperature_K = 10.0 ** float(log_tmax)
        self._quadrature_points = int(quadrature_points)
        self._mineps_offset = float(mineps_offset)

    def __call__(self, temperatures_keV, spectra):
        temperatures = _float64_vector(temperatures_keV, "AtomDB temperatures")
        spectra = np.asarray(spectra, dtype=np.float64)
        if spectra.ndim != 2 or spectra.shape[0] != temperatures.size:
            raise ValueError("spectra must have shape (n_temperatures, n_bins)")
        if not np.isfinite(spectra).all() or np.any(spectra < 0.0):
            raise ValueError("AtomDB spectra must be finite and nonnegative")
        spectra = np.ascontiguousarray(spectra)
        if self._parameters.size == 0:
            parameters = np.empty(1, dtype=np.float64)
        else:
            parameters = self._parameters
        output = np.empty(spectra.shape[1], dtype=np.float64)
        error = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        status = self._function(
            temperatures.size,
            spectra.shape[1],
            temperatures.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            spectra.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._parameters.size,
            parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._normalization,
            self._minimum_temperature_K,
            self._maximum_temperature_K,
            self._quadrature_points,
            self._mineps_offset,
            self._dem_function,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            error,
            len(error),
        )
        if status:
            message = bytes(error.raw).split(b"\0", 1)[0].decode("utf-8", "replace")
            detail = f": {message}" if message else ""
            raise RuntimeError(
                f"native AtomDB DEM evaluation failed ({status}){detail}"
            )
        return output


class AtomDBDEMModel(ArithmeticModel):
    """Sherpa CIE model integrating local AtomDB emissivities over a DEM.

    The default power-law distribution is proportional to
    ``(T/Tmax)**gamma`` on ``[Tmin, Tmax]`` and is normalized so ``norm`` is
    the integrated APEC-style scaled volume emission measure. Temperature
    bounds are fitted in ``log10(K)``. PyAtomDB prepares single-temperature
    spectra once per Sherpa energy grid; all repeated interpolation, DEM
    evaluation, normalization, and integration are performed in native C++.
    ``abundset`` accepts the same compositions as AtomDB queries. The baseline
    and fixed element multipliers are copied at construction so cached spectra
    retain their composition across later database or XSPEC setting changes.

    Provide ``dem_source`` or ``dem_library`` to define a different native
    DEM shape. The native function returns a nonnegative shape; this model
    normalizes that shape to the total ``norm`` parameter over the fitted
    temperature bounds.
    """

    ndim = 1

    def __init__(
        self,
        name,
        atomdb,
        *,
        dem="powerlaw",
        dem_source=None,
        dem_library=None,
        parameters=None,
        normalization=1.e-3,
        log_tmin=6.0,
        log_tmax=8.0,
        abundances=None,
        abundset=None,
        quadrature_points=128,
        parameter_bounds=None,
        compiler=None,
        include_dirs=(),
        extra_compile_args=(),
        extra_link_args=(),
    ):
        if _SHERPA_IMPORT_ERROR is not None:
            raise ImportError("Sherpa is required to construct AtomDBDEMModel") \
                from _SHERPA_IMPORT_ERROR
        if not isinstance(atomdb, AtomDB):
            raise TypeError("atomdb must be an open sherpa_plasma.AtomDB instance")
        if dem not in ("powerlaw", "custom"):
            raise ValueError("dem must be 'powerlaw' or 'custom'")
        if dem == "custom" and (dem_source is None) == (dem_library is None):
            raise ValueError("custom DEM requires exactly one of dem_source or dem_library")
        if dem == "powerlaw" and (dem_source is not None or dem_library is not None):
            raise ValueError("power-law DEM does not accept a custom native function")
        if dem == "powerlaw" and (
            compiler is not None or include_dirs or extra_compile_args or extra_link_args
        ):
            raise ValueError("compile options are only valid with a custom DEM")
        if parameters is None:
            parameters = {"gamma": -2.0} if dem == "powerlaw" else {}
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping of DEM parameter names to values")
        if dem == "powerlaw" and set(parameters) != {"gamma"}:
            raise ValueError("power-law DEM parameters must contain only gamma")
        if parameter_bounds is None:
            parameter_bounds = {}
        if not isinstance(parameter_bounds, Mapping):
            raise TypeError("parameter_bounds must be a mapping")
        if set(parameter_bounds) - set(parameters):
            raise ValueError("parameter bounds were supplied for unknown DEM parameters")

        normalization = _positive(normalization, "normalization", zero=True)
        if normalization.ndim != 0:
            raise ValueError("normalization must be a scalar")
        normalization = float(normalization)
        log_tmin = float(log_tmin)
        log_tmax = float(log_tmax)
        if not math.isfinite(log_tmin) or not math.isfinite(log_tmax) or log_tmin >= log_tmax:
            raise ValueError("require finite log_tmin < log_tmax")
        if isinstance(quadrature_points, (bool, np.bool_)):
            raise ValueError("quadrature_points must be an integer >= 2")
        if int(quadrature_points) != quadrature_points or quadrature_points < 2:
            raise ValueError("quadrature_points must be an integer >= 2")

        self._atomdb = atomdb
        self._abundances = dict(abundances or {})
        self._abundset = _resolve_abundance_spec(
            atomdb.abundset if abundset is None else abundset)
        self._quadrature_points = int(quadrature_points)
        self._table_cache = OrderedDict()
        self._mineps_offset = 1.e-40

        tmin = _Parameter(
            name, "logTmin", log_tmin, min=log_tmin - 1., max=log_tmax - 1.e-6
        )
        tmax = _Parameter(
            name, "logTmax", log_tmax, min=log_tmin + 1.e-6, max=log_tmax + 1.
        )
        norm = _Parameter(name, "norm", normalization, min=0.)
        self.logTmin = tmin
        self.logTmax = tmax
        self.norm = norm
        model_parameters = [norm, tmin, tmax]
        for parameter_name, raw_value in parameters.items():
            if (not isinstance(parameter_name, str) or
                    not parameter_name.isidentifier() or
                    parameter_name.startswith("_") or
                    parameter_name.lower() in {
                        item.lower() for item in _RESERVED_PARAMETER_NAMES
                    } | {"norm", "logtmin", "logtmax"}):
                raise ValueError(f"invalid or reserved DEM parameter name: {parameter_name!r}")
            if isinstance(raw_value, (bool, np.bool_)):
                raise ValueError(f"initial value for {parameter_name} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"initial value for {parameter_name} must be finite")
            if parameter_name in parameter_bounds:
                bounds = tuple(map(float, parameter_bounds[parameter_name]))
                if len(bounds) != 2 or not bounds[0] < bounds[1]:
                    raise ValueError(
                        f"bounds for {parameter_name} must satisfy min < max"
                    )
                if not bounds[0] <= value <= bounds[1]:
                    raise ValueError(
                        f"initial value for {parameter_name} is outside its bounds"
                    )
                parameter = _Parameter(
                    name, parameter_name, value, min=bounds[0], max=bounds[1]
                )
            else:
                parameter = _Parameter(name, parameter_name, value)
            setattr(self, parameter_name, parameter)
            model_parameters.append(parameter)

        source = dem_source
        library = dem_library
        if dem == "powerlaw":
            source = Path(__file__).with_name("powerlaw_dem.cpp")
            library = None
        self._native_dem = _NativeDEM(
            source=source,
            library=library,
            compiler=compiler if dem == "custom" else None,
            include_dirs=include_dirs if dem == "custom" else (),
            extra_compile_args=extra_compile_args if dem == "custom" else (),
            extra_link_args=extra_link_args if dem == "custom" else (),
        )
        api = self._atomdb._using
        with api() as pyatomdb:
            self._mineps_offset = float(pyatomdb.const.MINEPSOFFSET)
        self._engine = None
        super().__init__(name, tuple(model_parameters))

    def _temperature_bounds(self, p):
        log_min, log_max = float(p[1]), float(p[2])
        if not math.isfinite(log_min) or not math.isfinite(log_max) or log_min >= log_max:
            raise ValueError("model evaluation requires logTmin < logTmax")
        lo, hi = self._atomdb.temperature_range
        log_minimum = math.log10(lo * _KEV_TO_K)
        log_maximum = math.log10(hi * _KEV_TO_K)
        if log_min < log_minimum or log_max > log_maximum:
            raise ValueError(
                f"DEM temperatures must lie in the local AtomDB range "
                f"[{lo * _KEV_TO_K:g}, {hi * _KEV_TO_K:g}] K"
            )
        return log_min, log_max

    @staticmethod
    def _energy_grid(xlo, xhi):
        lo = _float64_vector(xlo, "energy-bin lower edges")
        hi = _float64_vector(xhi, "energy-bin upper edges")
        if lo.size != hi.size or not lo.size:
            raise ValueError("energy-bin edges must be matching, nonempty arrays")
        if np.any(lo <= 0.0) or np.any(hi <= lo):
            raise ValueError("energy bins must satisfy 0 < lo < hi")
        if lo.size > 1 and not np.allclose(lo[1:], hi[:-1], rtol=1.e-12, atol=0.):
            raise ValueError("AtomDBDEMModel requires contiguous energy bins")
        return lo, hi, np.concatenate((lo[:1], hi))

    def _prepare_spectra(self, lo, hi, edges):
        key = (lo.tobytes(), hi.tobytes())
        if key in self._table_cache:
            self._table_cache.move_to_end(key)
            return self._table_cache[key]

        session = self._atomdb._cie()
        temperatures = np.asarray(session.spectra.kTlist, dtype=np.float64)
        if temperatures.ndim != 1 or temperatures.size < 2:
            raise ValueError("local AtomDB CIE table must contain at least two temperatures")
        spectra = np.empty((temperatures.size, lo.size), dtype=np.float64)
        with self._atomdb._using():
            self._atomdb._configure_session(
                session, self._abundances, abundset=self._abundset
            )
            session.set_response(edges, raw=True)
            for index, temperature in enumerate(temperatures):
                result = np.asarray(
                    session.return_spectrum(
                        float(temperature), teunit="keV", nearest=True
                    ),
                    dtype=np.float64,
                )
                if result.shape != (lo.size,) or not np.isfinite(result).all():
                    raise ValueError(
                        f"PyAtomDB returned an invalid spectrum at kT={temperature:g} keV"
                    )
                if np.any(result < 0.0):
                    raise ValueError(
                        f"PyAtomDB returned negative emissivity at kT={temperature:g} keV"
                    )
                spectra[index] = result
        table = (
            np.ascontiguousarray(temperatures),
            np.ascontiguousarray(spectra),
        )
        self._table_cache[key] = table
        if len(self._table_cache) > 4:
            self._table_cache.popitem(last=False)
        return table

    def calc(self, p, xlo, xhi=None, *args, **kwargs):
        if xhi is None:
            raise ValueError("AtomDBDEMModel requires low and high energy-bin edges")
        if args or set(kwargs) - {"integrate"}:
            raise TypeError("unexpected model evaluation arguments")
        integrate = kwargs.get("integrate", True)
        if not isinstance(integrate, (bool, np.bool_)) or not integrate:
            raise ValueError("AtomDBDEMModel supports integrated energy bins only")
        lo, hi, edges = self._energy_grid(xlo, xhi)
        log_min, log_max = self._temperature_bounds(p)
        norm = float(p[0])
        if not math.isfinite(norm) or norm < 0.0:
            raise ValueError("norm must be finite and nonnegative")
        temperatures, spectra = self._prepare_spectra(lo, hi, edges)
        self._engine = _NativeIntegrator(
            self._native_dem,
            parameters=p[3:],
            normalization=norm,
            log_tmin=log_min,
            log_tmax=log_max,
            quadrature_points=self._quadrature_points,
            mineps_offset=self._mineps_offset,
        )
        return self._engine(temperatures, spectra)

    def clear_cache(self):
        """Discard prepared spectra so the next evaluation rereads the response grid."""
        self._table_cache.clear()

