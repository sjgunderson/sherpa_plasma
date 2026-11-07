// Snapshot and restore XSPEC's DEM metadata around a native evaluation.
// Resolve the exported metadata from the XSPEC library already loaded by
// Sherpa; do not link or load a second XSPEC library.
#include <Python.h>
#include <XSFunctions/Utilities/FunctionUtility.h>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Values = std::vector<double>;

struct Library {
    void* handle;

    explicit Library(const char* path)
        : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        if (!handle) {
            const char* error = dlerror();
            throw std::runtime_error(error ? error : "dlopen failed");
        }
    }

    ~Library() {
        if (handle) dlclose(handle);
    }

    template<class T>
    T symbol(const char* name) {
        dlerror();
        void* ptr = dlsym(handle, name);
        const char* error = dlerror();
        if (error || !ptr) {
            throw std::runtime_error(
                std::string("XSPEC does not export ") + name);
        }
        return reinterpret_cast<T>(ptr);
    }
};

PyObject* array_list(const Values& values) {
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (!result) return nullptr;

    for (size_t i = 0; i < values.size(); ++i) {
        PyObject* item = PyFloat_FromDouble(values[i]);
        if (!item) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), item);
    }
    return result;
}

PyObject* arrays_tuple(const Values& temperatures, const Values& weights) {
    PyObject* t = array_list(temperatures);
    PyObject* w = array_list(weights);

    if (!t || !w) {
        Py_XDECREF(t);
        Py_XDECREF(w);
        return nullptr;
    }

    PyObject* result = PyTuple_Pack(2, t, w);
    Py_DECREF(t);
    Py_DECREF(w);
    return result;
}

// The FunctionUtility accessors in the installed header are inline. Resolve
// their exported backing data instead. nm displays an extra leading
// underscore for Mach-O symbols; dlsym takes the Itanium name shown here.
struct Metadata {
    Values* temperatures;
    Values* weights;
};

Metadata get_metadata(Library& library) {
    auto temperatures = library.symbol<Values*>(
        "_ZN15FunctionUtility10s_tempsDEME");
    auto weights = library.symbol<Values*>(
        "_ZN15FunctionUtility5s_DEME");

    if (!temperatures || !weights) {
        throw std::runtime_error("XSPEC DEM metadata symbols resolved to null");
    }

    return {temperatures, weights};
}

PyObject* snapshot(PyObject*, PyObject* args) {
    const char* path;
    if (!PyArg_ParseTuple(args, "s:snapshot", &path)) return nullptr;

    try {
        Library library(path);
        Metadata metadata = get_metadata(library);

        // Return detached copies, not views into XSPEC's shared state.
        const Values temperatures = *metadata.temperatures;
        const Values weights = *metadata.weights;
        return arrays_tuple(temperatures, weights);
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, error.what());
        }
        return nullptr;
    }
}

PyObject* capture(PyObject*, PyObject* args) {
    const char* path;
    PyObject* evaluate;

    if (!PyArg_ParseTuple(args, "sO:capture", &path, &evaluate)) {
        return nullptr;
    }
    if (!PyCallable_Check(evaluate)) {
        PyErr_SetString(PyExc_TypeError, "evaluate must be callable");
        return nullptr;
    }

    try {
        Library library(path);
        Metadata metadata = get_metadata(library);

        // Swap-based restoration avoids allocating in the cleanup path.
        struct Restore {
            Values* temperatures;
            Values* weights;
            Values old_temperatures;
            Values old_weights;

            ~Restore() {
                temperatures->swap(old_temperatures);
                weights->swap(old_weights);
            }
        } restore{
            metadata.temperatures,
            metadata.weights,
            *metadata.temperatures,
            *metadata.weights
        };

        // Clear first so a callback that does no native calculation cannot
        // pass by leaving old XSPEC metadata in place.
        metadata.temperatures->clear();
        metadata.weights->clear();

        PyObject* flux = PyObject_CallObject(evaluate, nullptr);
        if (!flux) return nullptr;
        Py_DECREF(flux);

        const Values temperatures = *metadata.temperatures;
        const Values weights = *metadata.weights;

        if (temperatures.empty() || temperatures.size() != weights.size()) {
            throw std::runtime_error(
                "Native evaluation did not publish fresh temperature weights; "
                "this component/build is unsupported or does not share XSPEC "
                "state with Sherpa");
        }

        return arrays_tuple(temperatures, weights);
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, error.what());
        }
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"capture", capture, METH_VARARGS,
     "Capture fresh native DEM arrays and restore previous DEM metadata."},
    {"snapshot", snapshot, METH_VARARGS,
     "Read detached XSPEC DEM metadata."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_xspec_bridge",
    nullptr,
    -1,
    methods
};
}

PyMODINIT_FUNC PyInit__xspec_bridge() {
    return PyModule_Create(&module);
}