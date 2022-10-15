#define PY_SSIZE_T_CLEAN 
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <stdio.h>
#include "spkmeans.h"

PyObject* fit_api(PyObject *self, PyObject *args);
PyObject* jacobian_api(PyObject *self, PyObject *args);
PyObject* weighted_api(PyObject *self, PyObject *args);
PyObject* diagonal_api(PyObject *self, PyObject *args);
PyObject* lnorm_api(PyObject *self, PyObject *args);
PyObject* eigen_api(PyObject *self, PyObject *args);
double ** to_matrix(PyArrayObject * obj);

static PyMethodDef capiMethods [] = {
    {"fit",(PyCFunction) fit_api, METH_VARARGS,PyDoc_STR("C kmeans")},
    {"to_jacobian",(PyCFunction) jacobian_api,METH_VARARGS,PyDoc_STR("jacobian")},
    {"to_weighted",(PyCFunction) weighted_api,METH_VARARGS,PyDoc_STR("weighted matrix")},
    {"to_diagonal",(PyCFunction) diagonal_api,METH_VARARGS,PyDoc_STR("diagonal matrix")},
    {"to_lnorm",(PyCFunction) lnorm_api,METH_VARARGS,PyDoc_STR("lnorm matrix")},
    {"eigengap",(PyCFunction) eigen_api,METH_VARARGS,PyDoc_STR("eigengap and normilized matrix")},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef moduledef  = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "Python interface for the kmeans C library function",
    -1,
    capiMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}



/* casting the Python Numpy Array to C array */
double ** to_matrix(PyArrayObject * obj){
    int i;
    double *pdata = (double*)PyArray_DATA(obj);
    int dimensional = PyArray_DIM(obj,1);
    int vec_counter = PyArray_DIM(obj,0);
    double **data = calloc(vec_counter, sizeof(double *));
    if (data == NULL){
        return NULL;
    }
    for(i=0; i<vec_counter ; i++){
        data[i] = pdata+i*dimensional;
    }
    return data;
}


/*------------------------------------------------------------------------------
Functions which connect the python functions to the C function in spkmeans.c
and convert the PyObjects to objects which suitable for the C function 
------------------------------------------------------------------------------*/

PyObject* jacobian_api(PyObject *self, PyObject *args){
    PyObject* maindata;
    PyObject* mainjacobi;
    double ** data;
    double ** jacobi;
    int result;
    int vec_counter;
    if (!PyArg_ParseTuple(args,"OOi",&maindata,&mainjacobi,&vec_counter)){
        return Py_BuildValue("i",1);
    }
    data = to_matrix((PyArrayObject *)maindata);
    jacobi = to_matrix((PyArrayObject *)mainjacobi);
    if (data == NULL || jacobi == NULL){
        if(data != NULL){
            free(data);
        }
        if(jacobi != NULL){
            free(jacobi);
        }
        return Py_BuildValue("i",1);
    }
    result = to_jacobian(jacobi,data,vec_counter);
    free(data);
    free(jacobi);
    return Py_BuildValue("i",result);
}


PyObject* weighted_api(PyObject *self, PyObject *args){
    PyObject* maindata;
    PyObject* mainweighted;
    double ** data;
    double ** weighted;
    int vec_counter;
    int dim;
    if (!PyArg_ParseTuple(args,"OOii",&maindata,&mainweighted,&vec_counter,&dim)){
        return Py_BuildValue("i",1);
    }
    data = to_matrix((PyArrayObject *)maindata);
    weighted = to_matrix((PyArrayObject *)mainweighted);
    if (data == NULL || weighted == NULL){
        if(data != NULL){
            free(data);
        }
        if(weighted != NULL){
            free(weighted);
        }
        return Py_BuildValue("i",1);
    }
    to_weighted(weighted,data,dim,vec_counter);
    free(data);
    free(weighted);
    return Py_BuildValue("i",0);
}


PyObject* diagonal_api(PyObject *self, PyObject *args){
    PyObject* maindata;
    PyObject* maindiagonal;
    double ** data;
    double ** diagonal;
    int vec_counter;
    if (!PyArg_ParseTuple(args,"OOi",&maindata,&maindiagonal,&vec_counter)){
        return Py_BuildValue("i",1);
    }
    data = to_matrix((PyArrayObject *)maindata);
    diagonal = to_matrix((PyArrayObject *)maindiagonal);
    if (data == NULL || diagonal == NULL){
        if(data != NULL){
            free(data);
        }
        if(diagonal != NULL){
            free(diagonal);
        }
        return Py_BuildValue("i",1);
    }
    to_diagonal(diagonal,data,vec_counter);
    free(data);
    free(diagonal);
    return Py_BuildValue("i",0);
}


PyObject* lnorm_api(PyObject *self, PyObject *args){
    PyObject* mainweighted;
    PyObject* maindiagonal;
    PyObject* mainlnorm;
    double ** weighted;
    double ** diagonal;
    double ** lnorm;
    int vec_counter;
    if (!PyArg_ParseTuple(args,"OOOi",&mainweighted,&maindiagonal,&mainlnorm,&vec_counter)){
        return Py_BuildValue("i",1);
    }
    weighted = to_matrix((PyArrayObject *)mainweighted);
    diagonal = to_matrix((PyArrayObject *)maindiagonal);
    lnorm = to_matrix((PyArrayObject *)mainlnorm);
     if (weighted == NULL || diagonal == NULL ||lnorm == NULL){
        if(weighted != NULL){
            free(weighted);
        }
        if(diagonal != NULL){
            free(diagonal);
        }
        if(lnorm != NULL){
            free(lnorm);
        }
        return Py_BuildValue("i",1);
    }
    to_lnorm(lnorm,diagonal,weighted,vec_counter);
    free(weighted);
    free(diagonal);
    free(lnorm);
    return Py_BuildValue("i",0);
}


PyObject* eigen_api(PyObject *self, PyObject *args){
    PyObject* mainjacobi;
    PyObject* mainT;
    double ** jacobi;
    double ** T;
    int vec_counter;
    int k;
    if (!PyArg_ParseTuple(args,"OOii",&mainjacobi,&mainT,&vec_counter,&k)){
        return Py_BuildValue("i",-1);
    }
    jacobi = to_matrix((PyArrayObject *)mainjacobi);
    T = to_matrix((PyArrayObject *)mainT);
    if(jacobi == NULL || T ==NULL){
        if(jacobi != NULL){
            free(jacobi);
        }
        if(T != NULL){
            free(T);
        }
        return Py_BuildValue("i",-1);
    }
    k = eigengap(jacobi,T,vec_counter,k);
    free(jacobi);
    free(T);
    return Py_BuildValue("i",k);
}


PyObject* fit_api(PyObject *self, PyObject *args){
    PyObject* points_data;
    PyObject* centroids_data;
    double ** points;
    double ** centroids;
    double EPS;
    int max_iter;
    if (!PyArg_ParseTuple(args,"idOO",&max_iter,&EPS,&points_data,&centroids_data)){
        return Py_BuildValue("i",1);
    }
    points = to_matrix((PyArrayObject *)points_data);
    centroids = to_matrix((PyArrayObject *)centroids_data);
    if(points == NULL || centroids ==NULL){
        if(points != NULL){
            free(centroids);
        }
        if(centroids != NULL){
            free(points);
        }
        return Py_BuildValue("i",-1);
    }
    int dimensional = PyArray_DIM((PyArrayObject *)points_data,1);
    int vec_counter = PyArray_DIM((PyArrayObject *)points_data,0);
    int k = PyArray_DIM((PyArrayObject *)centroids_data,0);
    fit(points, centroids, k, dimensional, vec_counter);
    return Py_BuildValue("i",0);
}