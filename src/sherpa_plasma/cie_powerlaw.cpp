#include "cie_model_abi.h"

#include <cmath>
#include <cstdio>

extern "C" SHERPA_PLASMA_CIE_EXPORT int sherpa_plasma_cie_model_v1(
    size_t n_bins,
    const double *energy_lo_keV,
    const double *energy_hi_keV,
    size_t n_parameters,
    const double *parameters,
    size_t,
    const double *,
    double *integrated_flux,
    char *error_message,
    size_t error_message_capacity
) {
    if (n_parameters != 2) {
        if (error_message_capacity) {
            std::snprintf(error_message, error_message_capacity,
                          "expected amplitude and photon index");
        }
        return 1;
    }

    const double amplitude = parameters[0];
    const double photon_index = parameters[1];
    if (!(amplitude >= 0.0) || !(photon_index > 0.0)) {
        if (error_message_capacity) {
            std::snprintf(error_message, error_message_capacity,
                          "amplitude must be nonnegative and index positive");
        }
        return 2;
    }

    for (size_t i = 0; i < n_bins; ++i) {
        const double lo = energy_lo_keV[i];
        const double hi = energy_hi_keV[i];
        if (photon_index == 1.0) {
            integrated_flux[i] = amplitude * std::log(hi / lo);
        } else {
            const double exponent = 1.0 - photon_index;
            const double upper = std::pow(hi, exponent);
            const double lower = std::pow(lo, exponent);
            integrated_flux[i] = amplitude * (upper - lower) / exponent;
        }
    }
    return 0;
}
