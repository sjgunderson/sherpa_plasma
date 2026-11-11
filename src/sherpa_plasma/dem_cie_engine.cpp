#include "dem_model_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
const double KEV_TO_K = 1.160451812e7;

void set_error(char *buffer, size_t capacity, const char *message) {
    if (buffer && capacity) {
        std::snprintf(buffer, capacity, "%s", message);
    }
}

bool valid(const double value) {
    return std::isfinite(value);
}

void spectrum_at(
    double temperature_keV,
    size_t n_temperature,
    size_t n_bins,
    const double *temperatures,
    const double *spectra,
    double mineps_offset,
    double *output
) {
    const double *upper_ptr = std::upper_bound(
        temperatures, temperatures + n_temperature, temperature_keV
    );
    size_t upper = static_cast<size_t>(upper_ptr - temperatures);
    if (upper == 0) upper = 1;
    if (upper >= n_temperature) upper = n_temperature - 1;
    const size_t lower = upper - 1;
    const double log_t = std::log(temperature_keV);
    const double log_lo = std::log(temperatures[lower]);
    const double log_hi = std::log(temperatures[upper]);
    const double fraction = (log_t - log_lo) / (log_hi - log_lo);
    for (size_t bin = 0; bin < n_bins; ++bin) {
        const double low = spectra[lower * n_bins + bin] + mineps_offset;
        const double high = spectra[upper * n_bins + bin] + mineps_offset;
        const double result = std::exp(
            (1.0 - fraction) * std::log(low) + fraction * std::log(high)
        ) - mineps_offset;
        output[bin] = result > 0.0 ? result : 0.0;
    }
}
}

extern "C" SHERPA_PLASMA_DEM_EXPORT int sherpa_plasma_dem_cie_model_v1(
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
) {
    if (!dem_function || !temperatures_keV || !spectra || !integrated_flux) {
        set_error(error_message, error_message_capacity, "null native input");
        return 1;
    }
    if (n_temperature < 2 || n_bins == 0 || n_quadrature == 0) {
        set_error(error_message, error_message_capacity,
                  "temperature table, energy bins, and quadrature must be nonempty");
        return 2;
    }
    if (!valid(normalization) || normalization < 0.0 ||
        !valid(minimum_temperature_K) || !valid(maximum_temperature_K) ||
        minimum_temperature_K <= 0.0 ||
        maximum_temperature_K <= minimum_temperature_K ||
        !valid(mineps_offset) || mineps_offset <= 0.0) {
        set_error(error_message, error_message_capacity,
                  "invalid normalization, temperature bounds, or interpolation offset");
        return 3;
    }

    const double minimum_keV = minimum_temperature_K / KEV_TO_K;
    const double maximum_keV = maximum_temperature_K / KEV_TO_K;
    const double temperature_tolerance = 1.e-12;
    if (minimum_keV < temperatures_keV[0] * (1.0 - temperature_tolerance) ||
        maximum_keV > temperatures_keV[n_temperature - 1] *
            (1.0 + temperature_tolerance)) {
        set_error(error_message, error_message_capacity,
                  "temperature bounds exceed the prepared AtomDB grid");
        return 4;
    }
    for (size_t i = 0; i < n_temperature; ++i) {
        if (!valid(temperatures_keV[i]) || temperatures_keV[i] <= 0.0 ||
            (i && temperatures_keV[i] <= temperatures_keV[i - 1])) {
            set_error(error_message, error_message_capacity,
                      "AtomDB temperatures must be positive and increasing");
            return 5;
        }
    }

    std::vector<double> numerator(n_bins, 0.0);
    std::vector<double> interpolated_spectrum(n_bins, 0.0);
    double denominator = 0.0;
    const double log_minimum = std::log(minimum_temperature_K);
    const double log_span =
        std::log(maximum_temperature_K) - log_minimum;
    const double delta_log_t = log_span / static_cast<double>(n_quadrature);
    char dem_error[512] = {0};

    for (size_t i = 0; i < n_quadrature; ++i) {
        const double temperature_K = std::exp(
            log_minimum + (static_cast<double>(i) + 0.5) * delta_log_t
        );
        double shape = 0.0;
        const int status = dem_function(
            temperature_K,
            minimum_temperature_K,
            maximum_temperature_K,
            n_parameters,
            parameters,
            &shape,
            dem_error,
            sizeof(dem_error)
        );
        if (status) {
            set_error(error_message, error_message_capacity,
                      dem_error[0] ? dem_error : "native DEM function failed");
            return 10 + status;
        }
        if (!valid(shape) || shape < 0.0) {
            set_error(error_message, error_message_capacity,
                      "native DEM function returned a nonfinite or negative shape");
            return 6;
        }

        const double weight = shape * temperature_K * delta_log_t;
        if (!valid(weight)) {
            set_error(error_message, error_message_capacity,
                      "native DEM integration overflowed");
            return 7;
        }
        denominator += weight;
        const double temperature_keV = temperature_K / KEV_TO_K;
        spectrum_at(
            temperature_keV, n_temperature, n_bins, temperatures_keV, spectra,
            mineps_offset, interpolated_spectrum.data()
        );
        for (size_t bin = 0; bin < n_bins; ++bin) {
            numerator[bin] += weight * interpolated_spectrum[bin];
        }
    }

    if (!valid(denominator) || denominator <= 0.0) {
        set_error(error_message, error_message_capacity,
                  "native DEM has zero integrated weight over the temperature range");
        return 8;
    }
    for (size_t bin = 0; bin < n_bins; ++bin) {
        integrated_flux[bin] = normalization * numerator[bin] / denominator;
        if (!valid(integrated_flux[bin]) || integrated_flux[bin] < 0.0) {
            set_error(error_message, error_message_capacity,
                      "native DEM integration returned invalid flux");
            return 9;
        }
    }
    return 0;
}
