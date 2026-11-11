# Native CIE models

This package provides two related interfaces:

* [`AtomDBDEMModel`](#atomdb-backed-dem-models) prepares local AtomDB CIE
  spectra and integrates them over a user-defined differential emission
  measure (DEM). This is the recommended path for custom thermal plasmas.
* [`CompiledCIEModel`](#write-a-native-function) is a lower-level interface
  where the user supplies the complete spectral model kernel.

`CompiledCIEModel` adapts a user-supplied native C or C++ function to Sherpa's
integrated source-model interface. Sherpa passes the energy-bin edges and
parameter values to one compiled function call; the function fills a contiguous
output array with one bin-integrated photon flux per bin. No Python callback is
run for individual bins.

The user-provided function defines the model physics. Python is used to set
parameter names, initial values, and optional bounds. This interface does not
implement a built-in AtomDB spectrum or read AtomDB FITS tables from C++;
`AtomDB` remains available for local database access and preprocessing. NEI
models are not supported by this interface.

The design follows ISIS's native CIE execution boundary: its CIE emissivity
interpolation and binning are implemented in native code, and ISIS supports
loadable modules. This interface applies the same principle to a
user-supplied kernel; it does not port the ISIS/AtomDB spectrum algorithm.
See ISIS's [`db-cie.c`](https://github.com/houckj/isis/blob/bc8e15a9aeecace0679fb9a90c6e8cabc551cfe6/src/db-cie.c)
and [loadable-module notes](https://github.com/houckj/isis/blob/bc8e15a9aeecace0679fb9a90c6e8cabc551cfe6/modules/README).

## Write a native function

Implement the ABI declared in `sherpa_plasma/cie_model_abi.h`:

```cpp
#include "cie_model_abi.h"

extern "C" SHERPA_PLASMA_CIE_EXPORT int sherpa_plasma_cie_model_v1(
    size_t n_bins,
    const double *energy_lo_keV,
    const double *energy_hi_keV,
    size_t n_parameters,
    const double *parameters,
    size_t n_model_data,
    const double *model_data,
    double *integrated_flux,
    char *error_message,
    size_t error_message_capacity
);
```

Return zero on success and a nonzero status on failure. When possible, write
a null-terminated explanation to `error_message`. Energy edges are in keV.
`parameters` follow the insertion order of the Python mapping. Return finite,
nonnegative, **bin-integrated** flux values, not flux densities.
`model_data` is an optional C-order flattened `float64` buffer retained for
the lifetime of the model. Its shape and layout are defined by the
user-supplied kernel. This lets Python use the existing `AtomDB` APIs to load
or prepare table values once, then lets the native kernel read them directly
on each evaluation without per-bin Python callbacks or repeated list/tuple
conversion.

For example, prepare a temperature/emissivity table from a local AtomDB once
and pass its C-order buffer to a kernel that expects pairs of values:

```python
import numpy as np
from sherpa_plasma import AtomDB, LineID

with AtomDB() as db:
    rows = db.find_lines(21.5, 22.2, element="O", ion=7, kT=.2, brightest=1)
    line = LineID.from_row(rows[0])
    temperature = db.temperatures
    emissivity = db.line_emissivity(line, temperature)
    model_data = np.ascontiguousarray(
        np.column_stack((temperature, emissivity)), dtype=np.float64
    )
```

Pass `model_data=model_data` to a matching `CompiledCIEModel`; its native
function receives the flattened pairs and `n_model_data` on every call.

The package includes a simple power-law example in
[`cie_powerlaw.cpp`](../src/sherpa_plasma/cie_powerlaw.cpp). It illustrates
the calling convention; it is not an AtomDB thermal-plasma model.

## Compile a source file at runtime

Runtime compilation requires a C++ compiler (`c++` or `g++` by default) and is
supported on Linux and macOS:

```python
from pathlib import Path
import sherpa_plasma
from sherpa_plasma import CompiledCIEModel

model = CompiledCIEModel(
    "native_powerlaw",
    source=Path(sherpa_plasma.__file__).with_name("cie_powerlaw.cpp"),
    parameters={"amplitude": 1e-3, "photon_index": 2.0},
    parameter_bounds={"amplitude": (0.0, 1.0), "photon_index": (0.1, 5.0)},
)
```

Compilation happens once when the model is constructed. For repeated use,
build and load a shared library instead:

```python
from sherpa_plasma import CompiledCIEModel, compile_cie_model

library = compile_cie_model("my_model.cpp", "my_model.so")
model = CompiledCIEModel(
    "my_model",
    library=library,
    parameters={"kT": 1.0, "norm": 1e-3},
)
```

The input arrays are converted, if needed, to contiguous `float64` NumPy
arrays and passed to native code by pointer. No per-bin Python iteration or
list/tuple serialization is used. A nonzero native status, compiler/load
failure, invalid energy grid, nonfinite output, or negative output raises an
exception rather than returning a success-shaped fallback.

Use the model as a Sherpa source model and fit its parameters as usual. The
model is one-dimensional and accepts integrated energy bins only. Compile
native code only from sources you trust.

## AtomDB-backed DEM models

`AtomDBDEMModel` uses PyAtomDB to prepare the single-temperature CIE spectra
on the requested energy grid. Sherpa fit evaluations then pass contiguous
NumPy tables to a compiled C++ integrator; the DEM callback, temperature
interpolation, normalization, and integration are all native. The prepared
spectra are cached per energy grid, so repeated fits do not call Python once
per temperature or energy bin. This starts with CIE only; NEI DEMs are not
implemented.

The built-in power-law DEM is

```text
dEM/dT proportional to (T/Tmax)**gamma,  Tmin <= T <= Tmax
```

The model numerically normalizes the shape over the fitted bounds, so `norm`
is the total APEC-style scaled volume emission measure
`1e-14 * VEM / (4*pi*distance**2)`. `logTmin` and `logTmax` are base-10
temperatures in Kelvin, and `gamma` is fitted like any other Sherpa
parameter. The integration uses log-temperature midpoints and the `T` Jacobian
for `dT = T d(log T)`. Both temperature limits must remain inside the local
AtomDB CIE temperature range.

```python
import sherpa.astro.ui as ui
from sherpa_plasma import AtomDB, AtomDBDEMModel

db = AtomDB("/path/to/APED")
dem = AtomDBDEMModel(
    "plaw_dem",
    db,
    normalization=1e-3,
    log_tmin=6.0,
    log_tmax=8.0,
    parameters={"gamma": -2.0},
    parameter_bounds={"gamma": (-5.0, 3.0)},
    quadrature_points=128,
)
ui.set_source(dem)
```

Keep the `AtomDB` connection open while the Sherpa model is in use, and close
it after fitting (`db.close()`). Sherpa's ordinary source-model parameters
include `norm`, `logTmin`, `logTmax`, and the custom DEM parameters.

### Supply a custom native DEM

For a DEM shape other than the built-in power law, write a C or C++ function
using `dem_model_abi.h` and select `dem="custom"`. The callback returns a
finite, nonnegative shape value; its absolute scale is irrelevant because the
engine normalizes it to `norm`. The parameter pointer follows the insertion
order of the `parameters` mapping. The callback is evaluated inside the
native integration loop.

```cpp
#include "dem_model_abi.h"
#include <cmath>

extern "C" SHERPA_PLASMA_DEM_EXPORT int sherpa_plasma_dem_shape_v1(
    double temperature_K,
    double minimum_temperature_K,
    double maximum_temperature_K,
    size_t n_parameters,
    const double *parameters,
    double *dem_shape,
    char *error_message,
    size_t error_message_capacity
) {
    if (n_parameters != 1) return 1;
    const double beta = parameters[0];
    *dem_shape = std::pow(temperature_K / maximum_temperature_K, beta);
    return 0;
}
```

Compile and load the custom shape from Python:

```python
from sherpa_plasma import AtomDBDEMModel, compile_dem_function

library = compile_dem_function("my_dem.cpp", "my_dem.so")
dem = AtomDBDEMModel(
    "custom_dem",
    db,
    dem="custom",
    dem_library=library,
    parameters={"beta": -1.0},
    parameter_bounds={"beta": (-5.0, 3.0)},
)
```

Alternatively, pass `dem_source="my_dem.cpp"` to compile it when constructing
the model. The packaged header directory is added to the compiler include path.
Compile only native code from sources you trust. For users who need to define
the entire spectrum rather than just its temperature distribution, use
`CompiledCIEModel` instead.
