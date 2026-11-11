// Isolated line projections using the same XSPEC APEC implementation as Sherpa.
#include <Python.h>
#include <XSFunctions/Aped.h>
#include <XSFunctions/Utilities/FunctionUtility.h>
#include <XSUtil/Error/Error.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>

namespace {
struct LineEntry {
    size_t temperature;
    size_t line;
    int transition;
    double emissivity;
};

struct Profiles {
    Aped aped;
    int element;
    std::vector<LineEntry> entries;
    std::vector<double> forbidden;
};

struct RestoreDatabase {
    std::map<std::string, double>& current;
    std::map<std::string, double> saved;

    RestoreDatabase()
        : current(const_cast<std::map<std::string, double>&>(FunctionUtility::getAllDbValues())),
          saved(current) {}
    ~RestoreDatabase() { current.swap(saved); }
};

void destroy_profiles(PyObject* capsule) {
    auto* p = static_cast<Profiles*>(PyCapsule_GetPointer(capsule, "xspec.profiles"));
    delete p;
}

std::vector<double> numbers(PyObject* input, const char* message) {
    std::unique_ptr<PyObject, decltype(&Py_DecRef)> seq(
        PySequence_Fast(input, message), &Py_DecRef);
    if (!seq) throw std::runtime_error(message);
    std::vector<double> output;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    output.reserve(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        double value = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq.get(), i));
        if (PyErr_Occurred()) throw std::runtime_error(message);
        output.push_back(value);
    }
    return output;
}

PyObject* create(PyObject*, PyObject* args) {
    const char* library_path;
    const char* root;
    int element, ground;
    double minimum, maximum;
    PyObject* transitions;
    PyObject* wavelength_obj = nullptr;
    if (!PyArg_ParseTuple(args, "ssiiOdd|O:create", &library_path, &root,
                          &element, &ground, &transitions, &minimum, &maximum,
                          &wavelength_obj)) return nullptr;
    try {
        void* library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error("cannot open Sherpa XSPEC library");
        void* native = dlsym(library, "_Z13getAtomicMassRKi");
        bool matches = native && native == reinterpret_cast<void*>(&getAtomicMass);
        dlclose(library);
        if (!matches)
            throw std::runtime_error("profile bridge and Sherpa use different XSPEC libraries");

        auto upper = numbers(transitions, "invalid triplet transitions");
        if (element < 3 || element > 30 || ground < 1 || upper.size() != 3
                || !std::isfinite(minimum) || !std::isfinite(maximum)
                || minimum <= 0 || maximum < minimum)
            throw std::invalid_argument("invalid triplet identity");
        for (size_t j = 0; j < upper.size(); ++j)
            if (!std::isfinite(upper[j]) || upper[j] < 1 ||
                    std::floor(upper[j]) != upper[j] ||
                    (j && upper[j] == upper[j - 1]) ||
                    (j == 2 && upper[j] == upper[0]))
                throw std::invalid_argument("invalid triplet transitions");
        auto profiles = std::unique_ptr<Profiles>(new Profiles);
        profiles->element = element;
        // The interval arguments are retained for API compatibility. The native
        // database, not a per-ion complete-triplet interval, defines the grid.
        if (profiles->aped.Read(std::string(root) + "_coco.fits",
                                std::string(root) + "_line.fits") != 0)
            throw std::runtime_error("could not read APEC files in native profile bridge");
        // Read every native temperature. An absent transition contributes zero,
        // independently of the other two transitions at that temperature.
        const int count = profiles->aped.NumberTemperatures();
        std::vector<std::array<double, 3>> emissivity(count, {{0., 0., 0.}});
        std::array<double, 3> energies = {{0., 0., 0.}};
        ApedElementRecord prototype;
        bool have_prototype = false;
        for (int t = 0; t < count; ++t) {
            if (profiles->aped.ReadTemperature(t) != 0)
                throw std::runtime_error("could not read APEC temperature in profile bridge");
            int found[3] = {0, 0, 0};
            auto& elements = profiles->aped.m_TemperatureRecord[t].m_ElementRecord;
            for (auto& el : elements) {
                bool selected_ion = false;
                for (auto& ion : el.m_IonRecord) {
                    if (el.m_AtomicNumber == element && ion.m_Ion == element - 1) {
                        selected_ion = true;
                        for (size_t j = 0; j < ion.m_LineEmissivity.size(); ++j) {
                            if (ion.m_LowerLevel[j] != ground) continue;
                            for (int k = 0; k < 3; ++k) {
                                if (ion.m_UpperLevel[j] != upper[k]) continue;
                                if (++found[k] > 1)
                                    throw std::runtime_error("native APEC triplet is ambiguous");
                                emissivity[t][k] = ion.m_LineEmissivity[j];
                                double energy = ion.m_LineEnergy[j];
                                if (!std::isfinite(emissivity[t][k]) || emissivity[t][k] < 0
                                        || !std::isfinite(energy) || energy <= 0)
                                    throw std::runtime_error("invalid native APEC transition");
                                if (energies[k] && energies[k] != energy)
                                    throw std::runtime_error("native triplet line energy changes with temperature");
                                energies[k] = energy;
                            }
                        }
                    }
                    // Retain an empty continuum record for XSPEC's CIE lookup.
                    ion.m_LineEmissivity = 0.;
                    ion.m_ContinuumEnergy.resize(0);
                    ion.m_ContinuumFlux.resize(0);
                    ion.m_ContinuumFluxError.resize(0);
                    ion.m_PseudoContinuumEnergy.resize(0);
                    ion.m_PseudoContinuumFlux.resize(0);
                    ion.m_PseudoContinuumFluxError.resize(0);
                    if (el.m_AtomicNumber != element || ion.m_Ion != element - 1) {
                        ion.m_LineEnergy.resize(0);
                        ion.m_LineEnergyError.resize(0);
                        ion.m_LineEmissivity.resize(0);
                        ion.m_LineEmissivityError.resize(0);
                        ion.m_ElementDriver.clear();
                        ion.m_IonDriver.clear();
                        ion.m_UpperLevel.clear();
                        ion.m_LowerLevel.clear();
                    }
                }
                if (selected_ion && !have_prototype) {
                    prototype = el;
                    have_prototype = true;
                }
            }
        }
        if (!have_prototype)
            throw std::runtime_error("selected He-like ion not found in native APEC data");
        std::vector<double> wavelengths;
        if (wavelength_obj) wavelengths = numbers(wavelength_obj, "invalid triplet wavelengths");
        if (!wavelengths.empty() && wavelengths.size() != 3)
            throw std::invalid_argument("expected three triplet wavelengths");
        for (int k = 0; k < 3; ++k) {
            if (!energies[k] && wavelengths.size() == 3
                    && std::isfinite(wavelengths[k]) && wavelengths[k] > 0)
                energies[k] = 12.39841974 / wavelengths[k];
            if (!energies[k])
                throw std::runtime_error("unlisted transition requires matching atomic wavelength");
        }
        std::array<int, 3> order = {{0, 1, 2}};
        std::sort(order.begin(), order.end(), [&](int a, int b) { return energies[a] < energies[b]; });
        for (auto& ion : prototype.m_IonRecord) {
            if (ion.m_Ion != element - 1) continue;
            ion.m_LineEnergy.resize(3);
            ion.m_LineEnergyError.resize(3);
            ion.m_LineEmissivity.resize(3);
            ion.m_LineEmissivityError.resize(3);
            ion.m_ElementDriver.resize(3);
            ion.m_IonDriver.resize(3);
            ion.m_UpperLevel.resize(3);
            ion.m_LowerLevel.resize(3);
            for (size_t j = 0; j < 3; ++j) {
                int k = order[j];
                ion.m_LineEnergy[j] = energies[k];
                ion.m_LineEnergyError[j] = 0.;
                ion.m_LineEmissivity[j] = 0.;
                ion.m_LineEmissivityError[j] = 0.;
                ion.m_ElementDriver[j] = element;
                ion.m_IonDriver[j] = element - 1;
                ion.m_UpperLevel[j] = static_cast<int>(upper[k]);
                ion.m_LowerLevel[j] = ground;
            }
        }
        // Give each native temperature an explicit isolated triplet record.
        // Zero-emissivity rows allow pumping to populate a previously absent
        // intercombination line with a profile at the donor's temperature.
        profiles->forbidden.resize(count);
        for (int t = 0; t < count; ++t) {
            auto& elements = profiles->aped.m_TemperatureRecord[t].m_ElementRecord;
            elements.clear();
            elements.push_back(prototype);
            profiles->forbidden[t] = emissivity[t][0];
            for (size_t j = 0; j < 3; ++j) {
                int k = order[j];
                profiles->entries.push_back({static_cast<size_t>(t), j, k, emissivity[t][k]});
            }
        }
        PyObject* capsule = PyCapsule_New(profiles.get(), "xspec.profiles", destroy_profiles);
        if (!capsule) return nullptr;
        profiles.release();
        return capsule;
    } catch (const YellowAlert&) {
        PyErr_SetString(PyExc_RuntimeError, "XSPEC failed while loading isolated APEC profiles");
        return nullptr;
    } catch (const RedAlert&) {
        PyErr_SetString(PyExc_RuntimeError, "XSPEC reported a fatal error loading APEC profiles");
        return nullptr;
    } catch (const std::exception& e) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyObject* project_profiles(PyObject* args, bool transfer) {
    PyObject *capsule, *edge_obj, *temp_obj, *weight_obj;
    double redshift, velocity;
    if (!PyArg_ParseTuple(args, "OOOOdd:project", &capsule, &edge_obj, &temp_obj,
                          &weight_obj, &redshift, &velocity)) return nullptr;
    auto* profiles = static_cast<Profiles*>(PyCapsule_GetPointer(capsule, "xspec.profiles"));
    if (!profiles) return nullptr;
    try {
        auto edges = numbers(edge_obj, "invalid energy edges");
        auto temperatures = numbers(temp_obj, "invalid temperatures");
        auto weights = numbers(weight_obj, "invalid DEM weights");
        if (edges.size() < 3 || temperatures.empty() || temperatures.size() != weights.size()
                || !std::isfinite(redshift) || redshift <= -1
                || !std::isfinite(velocity) || velocity < 0)
            throw std::invalid_argument("invalid profile projection inputs");
        for (size_t j = 0; j < edges.size(); ++j)
            if (!std::isfinite(edges[j]) || edges[j] <= 0 ||
                    (j && edges[j] <= edges[j - 1]))
                throw std::invalid_argument("energy edges must be positive and increasing");
        for (size_t j = 0; j < temperatures.size(); ++j)
            if (!std::isfinite(temperatures[j]) ||
                    temperatures[j] <= 0 ||
                    !std::isfinite(weights[j]) || weights[j] < 0)
                throw std::invalid_argument("invalid DEM weights or temperatures");

        // XSPEC's temperature search reads past the grid at its exact upper endpoint.
        const auto& grid = profiles->aped.m_Temperatures;
        const double upper_edge = grid[grid.size() - 1];
        for (auto& temperature : temperatures) {
            temperature = std::max(grid[0], std::min(temperature, upper_edge));
            if (temperature == upper_edge)
                temperature = std::nextafter(temperature, 0.);
        }

        RealArray energy(edges.data(), edges.size());
        RealArray temp(temperatures.data(), temperatures.size());
        RealArray dem(weights.data(), weights.size());
        IntegerVector elements(1, profiles->element);
        RealArray abundance(1., 1);
        RestoreDatabase restore_database;
        profiles->aped.SetControlOptions(false, false, true, velocity, 0.,
                                         false, false, false, false, "APEC");
        std::vector<std::vector<double>> lines(3);
        for (int k = 0; k < 3; ++k) {
            for (const auto& entry : profiles->entries) {
                auto& els = profiles->aped.m_TemperatureRecord[entry.temperature].m_ElementRecord;
                for (auto& el : els)
                    if (el.m_AtomicNumber == profiles->element)
                        for (auto& ion : el.m_IonRecord)
                            if (ion.m_Ion == profiles->element - 1)
                                ion.m_LineEmissivity[entry.line] = entry.transition == k
                                    ? (transfer ? profiles->forbidden[entry.temperature]
                                                : entry.emissivity) : 0.;
            }
            RealArray flux(0., edges.size() - 1), error(0., edges.size() - 1);
            profiles->aped.SumEqSpectra(energy, elements, abundance, redshift,
                                         temp, dem, flux, error);
            lines[k].reserve(flux.size());
            for (size_t i = 0; i < flux.size(); ++i) {
                if (!std::isfinite(flux[i]) || flux[i] < 0)
                    throw std::runtime_error("native APEC returned an invalid line profile");
                lines[k].push_back(flux[i]);
            }
        }
        PyObject* result = PyTuple_New(3);
        if (!result) return nullptr;
        for (int k = 0; k < 3; ++k) {
            PyObject* line = PyList_New(static_cast<Py_ssize_t>(lines[k].size()));
            if (!line) { Py_DECREF(result); return nullptr; }
            for (size_t i = 0; i < lines[k].size(); ++i) {
                PyObject* value = PyFloat_FromDouble(lines[k][i]);
                if (!value) { Py_DECREF(line); Py_DECREF(result); return nullptr; }
                PyList_SET_ITEM(line, static_cast<Py_ssize_t>(i), value);
            }
            PyTuple_SET_ITEM(result, k, line);
        }
        return result;
    } catch (const YellowAlert&) {
        PyErr_SetString(PyExc_RuntimeError, "XSPEC failed while projecting isolated APEC lines");
        return nullptr;
    } catch (const RedAlert&) {
        PyErr_SetString(PyExc_RuntimeError, "XSPEC reported a fatal error projecting APEC lines");
        return nullptr;
    } catch (const std::exception& e) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyObject* project(PyObject*, PyObject* args) { return project_profiles(args, false); }
PyObject* project_transfer(PyObject*, PyObject* args) { return project_profiles(args, true); }

PyObject* snapshot_db(PyObject*, PyObject*) {
    PyObject* result = PyDict_New();
    if (!result) return nullptr;
    for (const auto& entry : FunctionUtility::getAllDbValues()) {
        PyObject* value = PyFloat_FromDouble(entry.second);
        if (!value) { Py_DECREF(result); return nullptr; }
        int status = PyDict_SetItemString(result, entry.first.c_str(), value);
        Py_DECREF(value);
        if (status < 0) { Py_DECREF(result); return nullptr; }
    }
    return result;
}

PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Load isolated APEC triplet profiles."},
    {"project", project, METH_VARARGS, "Project the isolated triplet with XSPEC."},
    {"project_transfer", project_transfer, METH_VARARGS, "Project forbidden-donor profiles at each triplet energy."},
    {"_snapshot_db", snapshot_db, METH_NOARGS, "Read XSPEC diagnostic values."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_xspec_profiles", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__xspec_profiles() { return PyModule_Create(&module); }

