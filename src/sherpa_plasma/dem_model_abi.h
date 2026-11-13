#ifndef SHERPA_PLASMA_DEM_MODEL_ABI_H
#define SHERPA_PLASMA_DEM_MODEL_ABI_H

#include <stddef.h>

#if defined(_WIN32)
#define SHERPA_PLASMA_DEM_EXPORT __declspec(dllexport)
#else
#define SHERPA_PLASMA_DEM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*sherpa_plasma_dem_function_v1)(
    double temperature_K,
    double minimum_temperature_K,
    double maximum_temperature_K,
    size_t n_parameters,
    const double *parameters,
    double *dem_shape,
    char *error_message,
    size_t error_message_capacity
);

SHERPA_PLASMA_DEM_EXPORT int sherpa_plasma_dem_shape_v1(
    double temperature_K,
    double minimum_temperature_K,
    double maximum_temperature_K,
    size_t n_parameters,
    const double *parameters,
    double *dem_shape,
    char *error_message,
    size_t error_message_capacity
);

SHERPA_PLASMA_DEM_EXPORT int sherpa_plasma_dem_cie_model_v1(
    size_t n_temperature,
    size_t n_bins,
    const double *temperatures_keV,
    const double *spectra,
    size_t n_parameters,
    const double *parameters,
    double normalization,
    double minimum_temperature_K,
    double maximum_temperature_K,
    size_t n_quadrature,
    double mineps_offset,
    sherpa_plasma_dem_function_v1 dem_function,
    double *integrated_flux,
    char *error_message,
    size_t error_message_capacity
);

#ifdef __cplusplus
}
#endif

#endif
