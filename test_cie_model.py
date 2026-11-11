"""Tests for the native CIE model ABI and Sherpa-facing call contract."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from sherpa_plasma.cie_model import (
    CompiledCIEModel,
    _NativeKernel,
    compile_cie_model,
)


EXAMPLE_SOURCE = (
    Path(__file__).parents[1] / "src" / "sherpa_plasma" / "cie_powerlaw.cpp"
)
HAS_CPP_COMPILER = shutil.which("c++") is not None or shutil.which("g++") is not None
HAS_C_COMPILER = shutil.which("cc") is not None or shutil.which("gcc") is not None


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_compiled_source_evaluates_integrated_flux(tmp_path):
    library = compile_cie_model(EXAMPLE_SOURCE, tmp_path / "model.so")
    kernel = _NativeKernel(library, 2)
    lo = np.array([1., 2., 4.])
    hi = np.array([2., 4., 8.])

    actual = kernel(lo, hi, np.array([3., 2.]))

    np.testing.assert_allclose(actual, [1.5, .75, .375])


@pytest.mark.skipif(not HAS_C_COMPILER, reason="requires a C compiler")
def test_c_source_is_supported(tmp_path):
    source = tmp_path / "linear.c"
    source.write_text(
        """
        #include "cie_model_abi.h"
        int sherpa_plasma_cie_model_v1(
            size_t n, const double *lo, const double *hi, size_t npar,
            const double *p, size_t, const double *, double *out, char *error,
            size_t error_size) {
            if (npar != 1) return 1;
            for (size_t i = 0; i < n; ++i) out[i] = p[0] * (hi[i] - lo[i]);
            return 0;
        }
        """
    )
    library = compile_cie_model(source, tmp_path / "linear.so")
    kernel = _NativeKernel(library, 1)

    actual = kernel(np.array([1., 3.]), np.array([3., 4.]), np.array([2.]))

    np.testing.assert_allclose(actual, [4., 2.])


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_native_kernel_reports_model_errors(tmp_path):
    library = compile_cie_model(EXAMPLE_SOURCE, tmp_path / "model.so")
    kernel = _NativeKernel(library, 1)

    with pytest.raises(RuntimeError, match="expected amplitude and photon index"):
        kernel(np.array([1.]), np.array([2.]), np.array([3.]))


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_compiler_failure_includes_diagnostics(tmp_path):
    source = tmp_path / "broken.cpp"
    source.write_text("this is not valid C++")

    with pytest.raises(RuntimeError, match="failed to compile CIE model source"):
        compile_cie_model(source, tmp_path / "broken.so")


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_native_kernel_validates_arrays_and_native_output(tmp_path):
    library = compile_cie_model(EXAMPLE_SOURCE, tmp_path / "model.so")
    kernel = _NativeKernel(library, 2)

    with pytest.raises(ValueError, match="same length"):
        kernel(np.array([1.]), np.array([2., 3.]), np.array([1., 2.]))
    with pytest.raises(ValueError, match="0 < lo < hi"):
        kernel(np.array([2.]), np.array([1.]), np.array([1., 2.]))
    with pytest.raises(ValueError, match="finite"):
        kernel(np.array([1.]), np.array([2.]), np.array([np.nan, 2.]))


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_native_kernel_rejects_negative_output(tmp_path):
    source = tmp_path / "negative.cpp"
    source.write_text(
        """
        #include "cie_model_abi.h"
        extern "C" SHERPA_PLASMA_CIE_EXPORT int sherpa_plasma_cie_model_v1(
            size_t n, const double *, const double *, size_t, const double *,
            size_t, const double *, double *out, char *, size_t) {
            for (size_t i = 0; i < n; ++i) out[i] = -1.;
            return 0;
        }
        """
    )
    library = compile_cie_model(source, tmp_path / "negative.so")
    kernel = _NativeKernel(library, 0)

    with pytest.raises(ValueError, match="negative photon flux"):
        kernel(np.array([1.]), np.array([2.]), np.array([]))


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_model_data_is_passed_as_native_contiguous_buffer(tmp_path):
    source = tmp_path / "table.cpp"
    source.write_text(
        """
        #include "cie_model_abi.h"
        extern "C" SHERPA_PLASMA_CIE_EXPORT int sherpa_plasma_cie_model_v1(
            size_t n, const double *lo, const double *hi, size_t npar,
            const double *p, size_t ndata, const double *data, double *out,
            char *error, size_t error_size) {
            if (npar != 1 || ndata != 2) return 1;
            for (size_t i = 0; i < n; ++i)
                out[i] = p[0] * data[0] * (hi[i] - lo[i]) + data[1];
            return 0;
        }
        """
    )
    library = compile_cie_model(source, tmp_path / "table.so")
    table = np.array([[2., 3.]], dtype=np.float64)
    kernel = _NativeKernel(library, 1, model_data=table)

    actual = kernel(np.array([1.]), np.array([3.]), np.array([4.]))

    np.testing.assert_allclose(actual, [19.])
    assert kernel._model_data.flags.c_contiguous


@pytest.mark.skipif(not HAS_CPP_COMPILER, reason="requires a C++ compiler")
def test_sherpa_calc_uses_native_kernel_for_energy_bins(tmp_path):
    library = compile_cie_model(EXAMPLE_SOURCE, tmp_path / "model.so")
    model = object.__new__(CompiledCIEModel)
    model._native_kernel = _NativeKernel(library, 2)

    actual = model.calc(
        np.array([2., 2.]),
        np.array([1., 2.]),
        np.array([2., 4.]),
    )

    np.testing.assert_allclose(actual, [1., .5])
    with pytest.raises(ValueError, match="integrated bins only"):
        model.calc(
            np.array([2., 2.]),
            np.array([1., 2.]),
            np.array([2., 4.]),
            integrate=False,
        )


def test_model_requires_sherpa_when_constructed():
    from sherpa_plasma import cie_model

    if cie_model._SHERPA_IMPORT_ERROR is None:
        pytest.skip("Sherpa is installed")
    with pytest.raises(ImportError, match="Sherpa is required"):
        CompiledCIEModel("custom", parameters={"amplitude": 1.}, library="unused.so")
