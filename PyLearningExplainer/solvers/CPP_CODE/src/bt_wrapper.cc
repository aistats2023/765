
#ifndef CPP_CODE_BT_WRAPPER_H
#define CPP_CODE_BT_WRAPPER_H

#include<Python.h>
#include "ExplainerBT.h"

static PyObject* vectorToTuple_Int(const std::vector<int> &data) {
    PyObject* tuple = PyTuple_New( data.size() );
    if (!tuple) throw std::logic_error("Unable to allocate memory for Python tuple");
    for (unsigned int i = 0; i < data.size(); i++) {
        PyObject *num = PyLong_FromLong(data[i]);
        if (!num) {
            Py_DECREF(tuple);
            throw std::logic_error("Unable to allocate memory for Python tuple");
        }
        PyTuple_SET_ITEM(tuple, i, num);
    }
    return tuple;
}


static PyObject *void_to_pyobject(void *ptr) {
    return PyCapsule_New(ptr, NULL, NULL);
}

static void *pyobject_to_void(PyObject *obj) {
    return PyCapsule_GetPointer(obj, NULL);
}


PyObject *new_BT(PyObject *self, PyObject *args) {
    long val;
    if (!PyArg_ParseTuple(args, "L", &val))
        PyErr_Format(PyExc_TypeError, "The argument must be a integer representing the number of classes");
    //std::cout << "n_classes" << val << std::endl;
    
    PyLE::ExplainerBT *explainerBT = new PyLE::ExplainerBT(val);
    return void_to_pyobject(explainerBT);
}

static PyObject *add_tree(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *tree_obj;

    //std::cout << "add_tree" << std::endl;
    if (!PyArg_ParseTuple(args, "OO", &class_obj, &tree_obj))
        return NULL;

    if (!PyTuple_Check(tree_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple representing a raw tree and given by the python raw_tree() method !");
        return NULL;
    }

    // Get pointer to the class
    PyLE::ExplainerBT *explainerBT = (PyLE::ExplainerBT *) pyobject_to_void(class_obj);
    explainerBT->addTree(tree_obj);
    return Py_True;
}


static PyObject *compute_reason(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *vector_instance_obj;
    PyObject *vector_features_obj;

    long prediction;
    long n_iterations;
    long time_limit;
    long features_expressivity;
    if (!PyArg_ParseTuple(args, "OOOLLLL", &class_obj, &vector_instance_obj, &vector_features_obj, &prediction, &n_iterations, &time_limit, &features_expressivity))
        return NULL;

    if (!PyTuple_Check(vector_instance_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple reprenting the implicant !");
        return NULL;
    }

    if (!PyTuple_Check(vector_features_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The third argument must be a tuple represeting the features !");
        return NULL;
    }

    std::vector<int> reason;
    std::vector<int> instance;
    std::vector<int> features;

    // Convert the vector of the instance 
    Py_ssize_t size_obj = PyTuple_Size(vector_instance_obj);
    for(int i = 0; i < size_obj; i++) {
        PyObject *value_obj = PyTuple_GetItem(vector_instance_obj, i);
        instance.push_back(PyLong_AsLong(value_obj));
    }

    // Convert the vector of the features 
    size_obj = PyTuple_Size(vector_features_obj);
    for(int i = 0; i < size_obj; i++) {
        PyObject *value_obj = PyTuple_GetItem(vector_features_obj, i);
        features.push_back(PyLong_AsLong(value_obj));
    }

    // Get pointer to the class
    PyLE::ExplainerBT *explainerBT = (PyLE::ExplainerBT *) pyobject_to_void(class_obj);
    explainerBT->set_n_iterations(n_iterations);
    explainerBT->set_time_limit(time_limit);

    if (features_expressivity == 1)
      explainerBT->compute_reason_features(instance, features, prediction, reason);
    else
      explainerBT->compute_reason_conditions(instance, prediction, reason);


    return vectorToTuple_Int(reason);
}


// See https://gist.github.com/physacco/2e1b52415f3a964ad2a542a99bebed8f

// Method definition object for this extension, these arguments mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring

static PyMethodDef module_methods[] = {
        {"new_BT",         new_BT,         METH_VARARGS, "Create a BT object."},
        {"add_tree",       add_tree,       METH_VARARGS, "Add a tree."},
        {"compute_reason", compute_reason, METH_VARARGS, "Compute a reason"},
        {NULL,             NULL,           0,            NULL}
};


// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
        "c_explainer",
        "Explainer in C++ in order to improve performances.",
        -1,
        module_methods
};

// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_c_explainer(void) {
    Py_Initialize();
    return PyModule_Create(&module_definition);
}

#endif //CPP_CODE_BT_WRAPPER_H
