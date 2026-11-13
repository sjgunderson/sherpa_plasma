#include "dem_model_abi.h"

#include <cmath>
#include <cstdio>

extern "C" SHERPA_PLASMA_DEM_EXPORT int sherpa_plasma_dem_shape_v1(
    double temperature_K,
    double,
    double maximum_temperature_K,
    size_t n_parameters,
    const double *parameters,
    double *dem_shape,
    char *error_message,
    size_t error_message_capacity
) {
    if (n_parameters != 1) {
        if (error_message_capacity) {
            std::snprintf(error_message, error_message_capacity,
                          "power-law DEM requires one parameter: gamma");
        }
        return 1;
    }
    if (!std::isfinite(temperature_K) ||
        !std::isfinite(maximum_temperature_K) ||
        temperature_K <= 0.0 || maximum_temperature_K <= 0.0 ||
        !std::isfinite(parameters[0])) {
        if (error_message_capacity) {
            std::snprintf(error_message, error_message_capacity,
                          "power-law DEM received invalid temperature or gamma");
        }
        return 2;
    }

    *dem_shape = std::pow(temperature_K / maximum_temperature_K, parameters[0]);
    return 0;
}
