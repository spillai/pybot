// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#pragma once
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <Eigen/Eigen>
#include <glog/logging.h>

namespace pybot { namespace eigen_numpy {
void export_converters();
}  // namespace eigen_numpy
}  // namespace pybot



