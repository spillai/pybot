// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include <pybot/eigen_types.hpp>

namespace bot { namespace python {

static void py_init() {
  Py_Initialize();
  import_array();
}

static bool export_eigen_type_conversions = false;
bool init_and_export_eigen_converters() {

  if (export_eigen_type_conversions)
    return false;
  
  // std::cerr << "PYTHON TYPE CONVERTERS exported" << std::endl;
  export_eigen_type_conversions = true;

  // Py_Init and array import
  py_init();
  bot::eigen::export_converters();

  return true;
}

BOOST_PYTHON_MODULE(pybot_eigen_types)
{
  // Main types export
  bot::python::init_and_export_eigen_converters();
  py::scope scope = py::scope();
  
}

} // namespace python
} // namespace bot

