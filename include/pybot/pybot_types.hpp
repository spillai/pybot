// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#ifndef BOT_PYTHON_TYPES_HPP_
#define BOT_PYTHON_TYPES_HPP_

#include <assert.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "utils/template.h"
#include "utils/container.h"
#include "utils/pair.h"
#include "utils/optional.h"
// #include "utils/opencv_numpy_conversion.hpp"
// #include "utils/eigen_numpy_conversion.hpp"

namespace bot { namespace python {

bool init_and_export_converters();

} // namespace python
} // namespace bot

#endif // BOT_PYTHON_TYPES_HPP_
