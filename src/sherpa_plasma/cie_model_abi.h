#ifndef SHERPA_PLASMA_CIE_MODEL_ABI_H
#define SHERPA_PLASMA_CIE_MODEL_ABI_H

#include <stddef.h>

#if defined(_WIN32)
#define SHERPA_PLASMA_CIE_EXPORT __declspec(dllexport)
#else
#define SHERPA_PLASMA_CIE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

SHERPA_PLASMA_CIE_EXPORT int sherpa_plasma_cie_model_v1(
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

#ifdef __cplusplus
}
#endif

#endif
