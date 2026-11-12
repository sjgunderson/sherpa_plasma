// Controlled Aped interface for testing the production bridge without XSPEC.
// This tests bridge bookkeeping, not real APEC interpolation or line physics.
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <thread>
#include <valarray>
#include <vector>
#include "Utilities/FunctionUtility.h"

using RealArray = std::valarray<double>;
using IntegerVector = std::vector<int>;
double getAtomicMass(const int& element);

struct ApedIonRecord {
    int m_Ion = 0;
    RealArray m_ContinuumEnergy, m_ContinuumFlux, m_ContinuumFluxError;
    RealArray m_PseudoContinuumEnergy, m_PseudoContinuumFlux, m_PseudoContinuumFluxError;
    RealArray m_LineEnergy, m_LineEnergyError, m_LineEmissivity, m_LineEmissivityError;
    IntegerVector m_ElementDriver, m_IonDriver, m_UpperLevel, m_LowerLevel;
};
struct ApedElementRecord {
    int m_AtomicNumber = 14;
    std::vector<ApedIonRecord> m_IonRecord;
};
struct ApedTemperatureRecord {
    std::vector<ApedElementRecord> m_ElementRecord;
};
struct Aped {
    RealArray m_Temperatures = RealArray({.1, 1., 10.});
    std::vector<ApedTemperatureRecord> m_TemperatureRecord;
    std::string mode;
    double solar_scale = 1.;
    bool m_thermalBroadening = false, m_multiThread = false;
    double m_velocityBroadening = 0.;

    int Read(std::string, std::string lines) {
        mode = lines;
        if (mode.find("solar_zero") != std::string::npos) solar_scale = 0.;
        if (mode.find("solar_low") != std::string::npos) solar_scale = .25;
        if (mode.find("solar_high") != std::string::npos) solar_scale = 2.5;
        m_TemperatureRecord.resize(3);
        return 0;
    }
    int NumberTemperatures() { return 3; }
    int ReadTemperature(int t) {
        ApedElementRecord element;
        element.m_IonRecord.push_back(ApedIonRecord()); // CIE continuum
        ApedIonRecord ion;
        ion.m_Ion = 13;
        const double values[3][3] = {{2., 0., 0.}, {4., 2., 1.}, {1., 0., 0.}};
        std::vector<int> selected;
        for (int k = 0; k < 3; ++k)
            if (values[t][k] > 0 && !(k == 2 && mode.find("absent_i2") != std::string::npos))
                selected.push_back(k);
        if (t == 0 && mode.find("duplicate") != std::string::npos) selected.push_back(0);
        ion.m_LineEnergy.resize(selected.size());
        ion.m_LineEmissivity.resize(selected.size());
        ion.m_UpperLevel.resize(selected.size());
        ion.m_LowerLevel.resize(selected.size(), 1);
        const bool swapped = mode.find("swapped") != std::string::npos;
        const double energies[3] = {1.1, swapped ? 1.7 : 1.3, swapped ? 1.3 : 1.7};
        const int upper[3] = {2, 5, 6};
        for (size_t j = 0; j < selected.size(); ++j) {
            int k = selected[j];
            ion.m_LineEnergy[j] = energies[k];
            ion.m_LineEmissivity[j] = values[t][k];
            ion.m_UpperLevel[j] = upper[k];
        }
        element.m_IonRecord.push_back(ion);
        m_TemperatureRecord[t].m_ElementRecord.push_back(element);
        return 0;
    }
    bool option(const std::string& name, bool fallback) {
        const auto& settings = FunctionUtility::modelStrings();
        auto it = settings.find(name);
        if (it == settings.end() || it->second.empty()) return fallback;
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return value[0] == 'y' || value[0] == 't' || value.substr(0, 2) == "on";
    }
    void SetThermalBroadening(bool thermal, std::string type) {
        m_thermalBroadening = option(type + "THERMAL", thermal);
    }
    void SetMultiThread(bool threaded, std::string type) {
        m_multiThread = option(type + "MULTITHREAD", threaded);
    }
    void SetControlOptions(bool, bool, bool thermal, double velocity, double,
                           bool, bool, bool threaded, bool, std::string type) {
        SetThermalBroadening(thermal, type);
        SetMultiThread(threaded, type);
        m_velocityBroadening = velocity;
    }

    void SumEqSpectra(const RealArray& edges, const IntegerVector&, const RealArray&,
                     double redshift, const RealArray& temperatures, const RealArray& weights,
                     RealArray& flux, RealArray&) {
        flux = 0.;
        // Run independent calculations when requested, as native Aped does.
        // This is a controlled Gaussian implementation, not XSPEC physics.
        std::vector<RealArray> pieces(temperatures.size(), RealArray(0., flux.size()));
        auto calculate = [&](size_t i) {
            size_t low = 0, high = 0;
            double a = 0.;
            if (temperatures[i] >= m_Temperatures[2]) low = high = 2;
            else if (temperatures[i] > m_Temperatures[0]) {
                high = temperatures[i] < m_Temperatures[1] ? 1 : 2;
                low = high - 1;
                a = (temperatures[i] - m_Temperatures[low]) /
                    (m_Temperatures[high] - m_Temperatures[low]);
            }
            for (int side = 0; side < 2; ++side) {
                size_t t = side ? high : low;
                // Model the solar conversion that real Aped::SumEqSpectra does
                // internally; the Python wrapper must not apply it again.
                double coefficient = solar_scale * weights[i] * (side ? a : 1 - a);
                for (const auto& element : m_TemperatureRecord[t].m_ElementRecord)
                    for (const auto& ion : element.m_IonRecord)
                        for (size_t j = 0; j < ion.m_LineEnergy.size(); ++j) {
                            double energy = ion.m_LineEnergy[j] / (1 + redshift);
                            double variance = std::pow(m_velocityBroadening / 299792.458, 2);
                            if (m_thermalBroadening)
                                variance += m_Temperatures[t] * 1.60217733e-9 /
                                    (getAtomicMass(element.m_AtomicNumber) * 1.660539040e-24 *
                                     std::pow(299792.458e5, 2));
                            double sigma = energy * std::sqrt(variance);
                            for (size_t bin = 0; bin < flux.size(); ++bin) {
                                double fraction = sigma > 0 ? .5 * (
                                    std::erf((edges[bin + 1] - energy) / (std::sqrt(2.) * sigma)) -
                                    std::erf((edges[bin] - energy) / (std::sqrt(2.) * sigma))) :
                                    (edges[bin] <= energy && energy < edges[bin + 1] ? 1. : 0.);
                                pieces[i][bin] += fraction * coefficient * ion.m_LineEmissivity[j]
                                                  * 1.e14 / (1 + redshift);
                            }
                        }
            }
        };
        if (m_multiThread) {
            std::vector<std::thread> threads;
            for (size_t i = 0; i < temperatures.size(); ++i)
                threads.emplace_back(calculate, i);
            for (auto& thread : threads) thread.join();
        } else
            for (size_t i = 0; i < temperatures.size(); ++i) calculate(i);
        for (const auto& piece : pieces) flux += piece;
    }
};
