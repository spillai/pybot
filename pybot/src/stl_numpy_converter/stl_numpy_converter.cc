// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT
#include <iostream>
#include "stl_numpy_converter/utils/template.h"
#include "stl_numpy_converter/utils/container.h"
#include "stl_numpy_converter/stl_numpy_converter.h"

namespace bp = boost::python;

namespace stl_numpy {

static bool converter_exported = false;
void export_converters() {
  std::cout << "stl_numpy :: Registering converters for stl vectors" << std::endl;
  if (converter_exported) {
    std::cout << "stl_numpy :: Already registered" << std::endl;
    return;
  }

  Py_Initialize();
  // import_array();

  // vectors
  expose_template_type<int>();
  expose_template_type<float>();
  expose_template_type<double>();
  expose_template_type<unsigned int>();

  expose_template_type< std::vector<int> >();
  expose_template_type< std::vector<float> >();
  expose_template_type< std::vector<double> >();
  expose_template_type< std::vector<unsigned int> >();

  expose_template_type< std::pair<int, int> >();
  expose_template_type< std::vector<std::pair<int, int> > >();

  expose_template_type<std::map<int, std::vector<int> > >();
  expose_template_type<std::map<int, std::vector<float> > >();

  expose_template_type<std::map<std::string, float> >();
  expose_template_type<std::map<std::string, double> >();

  converter_exported = true;
}
}  // namespace stl_numpy
