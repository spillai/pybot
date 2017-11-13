// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#pragma once
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "opencv_numpy_converter/conversion.h"

namespace opencv_numpy {

namespace py = boost::python;

// TODO: Template these
// Vec3f => cv::Mat
struct Vec3f_to_mat {
  static PyObject* convert(const cv::Vec3f& v){
    NDArrayConverter cvt;
    PyObject* ret = cvt.toNDArray(cv::Mat(v));
    return ret;
  }
};

// cv::Point => cv::Mat
struct Point_to_mat {
  static PyObject* convert(const cv::Point& v){
    NDArrayConverter cvt;
    PyObject* ret = cvt.toNDArray(cv::Mat(v));
    return ret;
  }
};

// cv::Point2f => cv::Mat
struct Point2f_to_mat {
  static PyObject* convert(const cv::Point2f& v){
    NDArrayConverter cvt;
    PyObject* ret = cvt.toNDArray(cv::Mat(v));
    return ret;
  }
};

// cv::Point3f => cv::Mat
struct Point3f_to_mat {
  static PyObject* convert(const cv::Point3f& v){
    NDArrayConverter cvt;
    PyObject* ret = cvt.toNDArray(cv::Mat(v));
    return ret;
  }
};

// cv::Mat_<T> => Numpy PyObject
template <typename T>
struct Mat_to_PyObject {
  static PyObject* convert(const T& mat){
    NDArrayConverter cvt;
    PyObject* ret = cvt.toNDArray(mat);
    return ret;
  }
};

// Generic templated cv::Mat <=> Numpy PyObject converter
template <typename T>
struct Mat_PyObject_converter {
  // Register from converter
  Mat_PyObject_converter() {
    boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<T>());

    // Register to converter
    py::to_python_converter<T, Mat_to_PyObject<T> >();
  }

  // Convert from type T to PyObject (numpy array)
  // Assume obj_ptr can be converted in a cv::Mat
  static void* convertible(PyObject* obj_ptr) {
    // Check validity?
    assert(obj_ptr != 0);
    return obj_ptr;
  }

  // Convert obj_ptr into a cv::Mat
  static void construct(PyObject* obj_ptr,
                        boost::python::converter::rvalue_from_python_stage1_data* data) {
    namespace bp = boost::python;
    typedef bp::converter::rvalue_from_python_storage< T > storage_t;

    storage_t* the_storage = reinterpret_cast<storage_t*>(data);
    void* memory_chunk = the_storage->storage.bytes;

    NDArrayConverter cvt;
    T* newvec = new (memory_chunk) T(cvt.toMat(obj_ptr));
    data->convertible = memory_chunk;

    return;
  }
};

void export_converters();
void export_types();
}  // namespace opencv_numpy
