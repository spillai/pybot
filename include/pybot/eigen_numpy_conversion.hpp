// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#ifndef _EIGEN_NUMPY_H_
#define _EIGEN_NUMPY_H

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <Eigen/Eigen>
#include <glog/logging.h>

namespace bot { namespace eigen {
void export_converters();
} // namespace eigen
} // namespace bot
#endif


