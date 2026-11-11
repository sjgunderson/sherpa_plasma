// Controlled Aped interface for testing the production bridge without XSPEC.
// This tests bridge bookkeeping, not real APEC interpolation or line physics.
#pragma once
#include <algorithm>
#include <string>
#include <valarray>
#include <vector>

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

    int Read(std::string, std::string lines) {
        mode = lines;
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
    void SetControlOptions(bool, bool, bool, double, double, bool, bool, bool, bool, std::string) {}

    void SumEqSpectra(const RealArray& edges, const IntegerVector&, const RealArray&,
                     double redshift, const RealArray& temperatures, const RealArray& weights,
                     RealArray& flux, RealArray&) {
        flux = 0.;
        for (size_t i = 0; i < temperatures.size(); ++i) {
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
                double coefficient = weights[i] * (side ? a : 1 - a);
                for (const auto& element : m_TemperatureRecord[t].m_ElementRecord)
                    for (const auto& ion : element.m_IonRecord)
                        for (size_t j = 0; j < ion.m_LineEnergy.size(); ++j) {
                            double energy = ion.m_LineEnergy[j] / (1 + redshift);
                            for (size_t bin = 0; bin < flux.size(); ++bin)
                                if (edges[bin] <= energy && energy < edges[bin + 1])
                                    flux[bin] += coefficient * ion.m_LineEmissivity[j]
                                                 * 1.e14 / (1 + redshift);
                        }
            }
        }
    }
};
