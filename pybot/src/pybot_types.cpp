// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include "pybot_types.hpp"

namespace py = boost::python;

namespace pybot {

BOOST_PYTHON_MODULE(pybot_types)
{
  google::InitGoogleLogging("pybot");
  py::scope scope = py::scope();

  // Boost python initializations
  // Py_Initialize();
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

