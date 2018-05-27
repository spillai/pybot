// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "stl_numpy_converter/utils/template.h"
#include "stl_numpy_converter/utils/container.h"
#include "stl_numpy_converter/utils/optional.h"

#include "stl_numpy_converter/stl_numpy_converter.h"
#include "eigen_numpy_converter/eigen_numpy_converter.h"
#include "opencv_numpy_converter/opencv_numpy_converter.h"

namespace py = boost::python;

namespace pybot {

BOOST_PYTHON_MODULE(pybot_types)
{
  google::InitGoogleLogging("pybot");
  py::scope scope = py::scope();

  // Boost python initializations
  Py_Initialize();
  // import_array();

  // STL converters
  stl_numpy::export_converters();
  
  // Eigen-numpy converters
  eigen_numpy::export_converters();

  // OpenCV-numpy converters
  opencv_numpy::export_converters();
  opencv_numpy::export_types();
  
}

}  // namespace pybot

