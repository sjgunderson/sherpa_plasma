"""Sherpa source models backed by user-supplied native C/C++ kernels."""
from collections.abc import Mapping
import ctypes
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np

try:
    from sherpa.models.model import ArithmeticModel as _ArithmeticModel
    from sherpa.models.parameter import Parameter as _Parameter
except ImportError as exc:
    _SHERPA_IMPORT_ERROR = exc

    class _ArithmeticModel:
        pass

    _Parameter = None
else:
    _SHERPA_IMPORT_ERROR = None


_ABI_SYMBOL = "sherpa_plasma_cie_model_v1"
_ERROR_BUFFER_SIZE = 1024
_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
_RESERVED_PARAMETER_NAMES = {
    "calc", "close", "integrate", "name", "pars", "startup", "teardown",
}


def _source_path(source):
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"C/C++ model source does not exist: {path}")
    if path.suffix.lower() not in _SOURCE_SUFFIXES:
        raise ValueError(
            "model source must have a .c, .cc, .cpp, or .cxx extension"
        )
    return path


def _sequence_of_paths(values, name):
    if isinstance(values, (str, os.PathLike)):
        values = (values,)
    try:
        paths = tuple(Path(value).expanduser().resolve() for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a path or sequence of paths") from exc
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return paths


def _compiler_command(compiler, source):
    if compiler is None:
        variable = "CC" if source.suffix.lower() == ".c" else "CXX"
        default = ("cc" if shutil.which("cc") else "gcc") if variable == "CC" else (
            "c++" if shutil.which("c++") else "g++"
        )
        command = os.environ.get(variable, default)
        tokens = shlex.split(command)
    elif isinstance(compiler, os.PathLike):
        tokens = [os.fspath(compiler)]
    else:
        tokens = shlex.split(str(compiler))
    if not tokens:
        raise ValueError("compiler command must not be empty")
    executable = shutil.which(tokens[0])
    if executable is None:
        raise FileNotFoundError(f"compiler executable not found: {tokens[0]}")
    tokens[0] = executable
    return tokens


def _compile_shared_library(
    source,
    output,
    *,
    compiler=None,
    include_dirs=(),
    extra_compile_args=(),
    extra_link_args=(),
):
    source = _source_path(source)
    output = Path(output).expanduser().resolve()
    includes = (Path(__file__).resolve().parent,) + _sequence_of_paths(
        include_dirs, "include_dirs"
    )
    if sys.platform == "darwin":
        platform_flags = ["-dynamiclib", "-fPIC"]
    elif sys.platform.startswith("linux"):
        platform_flags = ["-shared", "-fPIC"]
    else:
        raise RuntimeError(
            "runtime C/C++ compilation currently supports Linux and macOS"
        )

    command = _compiler_command(compiler, source)
    command.extend(["-O3", *platform_flags])
    if source.suffix.lower() != ".c":
        command.append("-std=c++11")
    command.extend(f"-I{path}" for path in includes)
    command.extend(str(arg) for arg in extra_compile_args)
    command.extend([str(source), "-o", str(output)])
    command.extend(str(arg) for arg in extra_link_args)
    if source.suffix.lower() == ".c" and sys.platform != "darwin":
        command.append("-lm")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        diagnostic = "\n".join(
            text for text in (result.stdout.strip(), result.stderr.strip()) if text
        )
        if not diagnostic:
            diagnostic = "compiler returned no diagnostic output"
        raise RuntimeError(
            f"failed to compile CIE model source {source}:\n{diagnostic}"
        )
    if not output.is_file():
        raise RuntimeError(
            f"compiler succeeded but did not create the model library: {output}"
        )
    return output


def compile_cie_model(
    source,
    output,
    *,
    compiler=None,
    include_dirs=(),
    extra_compile_args=(),
    extra_link_args=(),
):
    """Compile a C/C++ model source into a shared library.

    The source must export ``sherpa_plasma_cie_model_v1`` as declared in
    ``cie_model_abi.h``. The packaged header directory is added to the include
    path automatically.
    """
    return _compile_shared_library(
        source,
        output,
        compiler=compiler,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


def _float64_vector(values, name):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _float64_model_data(values):
    if values is None:
        return np.empty(0, dtype=np.float64)
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("model_data must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError("model_data must contain only finite values")
    return np.ascontiguousarray(result.ravel(order="C"))


class _NativeKernel:
    def __init__(self, library_path, n_parameters, model_data=None):
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"compiled CIE model library does not exist: {path}")
        try:
            library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise RuntimeError(f"could not load CIE model library {path}: {exc}") from exc
        try:
            function = getattr(library, _ABI_SYMBOL)
        except AttributeError as exc:
            raise ValueError(
                f"CIE model library {path} does not export {_ABI_SYMBOL}"
            ) from exc

        double_pointer = ctypes.POINTER(ctypes.c_double)
        function.argtypes = [
            ctypes.c_size_t,
            double_pointer,
            double_pointer,
            ctypes.c_size_t,
            double_pointer,
            ctypes.c_size_t,
            double_pointer,
            double_pointer,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        self._library = library
        self._function = function
        self._n_parameters = n_parameters
        self._model_data = _float64_model_data(model_data)

    def __call__(self, energy_lo, energy_hi, parameters):
        lo = _float64_vector(energy_lo, "energy_lo")
        hi = _float64_vector(energy_hi, "energy_hi")
        pars = _float64_vector(parameters, "parameters")
        if lo.size != hi.size:
            raise ValueError("energy_lo and energy_hi must have the same length")
        if pars.size != self._n_parameters:
            raise ValueError(
                f"expected {self._n_parameters} model parameters, got {pars.size}"
            )
        if np.any(lo <= 0) or np.any(hi <= lo):
            raise ValueError("energy bins must have finite edges with 0 < lo < hi")
        if not lo.size:
            return np.empty(0, dtype=np.float64)

        result = np.empty(lo.size, dtype=np.float64)
        error = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        status = self._function(
            lo.size,
            lo.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            hi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pars.size,
            pars.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._model_data.size,
            self._model_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            error,
            len(error),
        )
        if status:
            message = bytes(error.raw).split(b"\0", 1)[0].decode("utf-8", "replace")
            detail = f": {message}" if message else ""
            raise RuntimeError(
                f"native CIE model returned status {status}{detail}"
            )
        if not np.isfinite(result).all():
            raise ValueError("native CIE model returned nonfinite flux")
        if np.any(result < 0):
            raise ValueError("native CIE model returned negative photon flux")
        return result


class CompiledCIEModel(_ArithmeticModel):
    """Sherpa model evaluated by a user-supplied compiled C/C++ kernel.

    Exactly one of ``source`` or ``library`` is required. ``source`` is
    compiled once when the model is constructed; ``library`` loads an existing
    shared library. Model parameters are supplied as a mapping of names to
    initial values, optionally with finite/infinite lower and upper bounds.
    ``calc`` passes contiguous float64 arrays directly to native code and
    expects one nonnegative, bin-integrated photon flux per energy bin.

    Python defines model parameters and configuration only; the per-bin
    calculation is always performed by the native function. NEI models are not
    currently supported.
    """

    ndim = 1

    def __init__(
        self,
        name,
        *,
        parameters,
        source=None,
        library=None,
        model_data=None,
        parameter_bounds=None,
        compiler=None,
        include_dirs=(),
        extra_compile_args=(),
        extra_link_args=(),
    ):
        if _SHERPA_IMPORT_ERROR is not None:
            raise ImportError(
                "Sherpa is required to construct CompiledCIEModel"
            ) from _SHERPA_IMPORT_ERROR
        if (source is None) == (library is None):
            raise ValueError("provide exactly one of source or library")
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError("parameters must be a nonempty name-to-value mapping")
        if parameter_bounds is None:
            parameter_bounds = {}
        if not isinstance(parameter_bounds, Mapping):
            raise TypeError("parameter_bounds must be a name-to-(min, max) mapping")

        parameter_objects = []
        initial_values = {}
        for parameter_name, raw_value in parameters.items():
            if (not isinstance(parameter_name, str) or
                    not parameter_name.isidentifier() or
                    parameter_name.startswith("_") or
                    parameter_name in _RESERVED_PARAMETER_NAMES):
                raise ValueError(
                    f"invalid Sherpa parameter name: {parameter_name!r}"
                )
            if isinstance(raw_value, (bool, np.bool_)):
                raise ValueError(f"initial value for {parameter_name} must be numeric")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"initial value for {parameter_name} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"initial value for {parameter_name} must be finite")

            if parameter_name in parameter_bounds:
                try:
                    bounds = tuple(parameter_bounds[parameter_name])
                except TypeError as exc:
                    raise ValueError(
                        f"bounds for {parameter_name} must contain (min, max)"
                    ) from exc
                if len(bounds) != 2:
                    raise ValueError(
                        f"bounds for {parameter_name} must contain (min, max)"
                    )
                lower, upper = map(float, bounds)
                if math.isnan(lower) or math.isnan(upper) or lower >= upper:
                    raise ValueError(
                        f"bounds for {parameter_name} must satisfy min < max"
                    )
                if not lower <= value <= upper:
                    raise ValueError(
                        f"initial value for {parameter_name} is outside its bounds"
                    )
                parameter = _Parameter(
                    name, parameter_name, value, min=lower, max=upper
                )
            else:
                parameter = _Parameter(name, parameter_name, value)
            setattr(self, parameter_name, parameter)
            parameter_objects.append(parameter)
            initial_values[parameter_name] = value
        unknown_bounds = set(parameter_bounds) - set(initial_values)
        if unknown_bounds:
            names = ", ".join(sorted(map(str, unknown_bounds)))
            raise ValueError(f"bounds supplied for unknown parameters: {names}")

        self._build_directory = None
        if source is not None:
            source = _source_path(source)
            self._build_directory = tempfile.TemporaryDirectory(
                prefix="sherpa-plasma-cie-"
            )
            output = Path(self._build_directory.name) / (
                "cie_model.dylib" if sys.platform == "darwin" else "cie_model.so"
            )
            try:
                library_path = _compile_shared_library(
                    source,
                    output,
                    compiler=compiler,
                    include_dirs=include_dirs,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                self._build_directory.cleanup()
                self._build_directory = None
                raise
        else:
            if (compiler is not None or include_dirs or extra_compile_args or
                    extra_link_args):
                raise ValueError(
                    "compiler and compile options are only valid with source"
                )
            library_path = library

        try:
            self._native_kernel = _NativeKernel(
                library_path, len(parameter_objects), model_data=model_data
            )
        except (OSError, RuntimeError, ValueError):
            if self._build_directory is not None:
                self._build_directory.cleanup()
                self._build_directory = None
            raise
        super().__init__(name, tuple(parameter_objects))

    def calc(self, p, xlo, xhi=None, *args, **kwargs):
        """Evaluate integrated photon flux for Sherpa's energy-bin edges."""
        if xhi is None:
            raise ValueError("CompiledCIEModel requires low and high energy-bin edges")
        if args or set(kwargs) - {"integrate"}:
            raise TypeError("unexpected model evaluation arguments")
        integrate = kwargs.get("integrate", True)
        if not isinstance(integrate, (bool, np.bool_)) or not integrate:
            raise ValueError("CompiledCIEModel supports integrated bins only")
        return self._native_kernel(xlo, xhi, p)
