// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include <pybot/eigen_types.hpp>

namespace pybot { 

// static void py_init() {
//   Py_Initialize();
//   import_array();
// }

// static bool eigen_types_registered = false;
// bool register_eigen_converters() {

//   if (eigen_types_registered)
//     return false;
  
//   // std::cerr << "PYTHON TYPE CONVERTERS exported" << std::endl;
//   eigen_types_registered = true;

//   // Py_Init and array import
//   py_init();
//   bot::eigen::export_converters();

//   return true;
// }

BOOST_PYTHON_MODULE(pybot_eigen_types)
{
  // Main types export
  py_init();
  pybot::eigen_numpy::export_converters();
  py::scope scope = py::scope();
  
}

} // namespace pybot

