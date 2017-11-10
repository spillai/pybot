// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include <pybot/pybot_types.hpp>

namespace bot { namespace python {

static void py_init() {
  Py_Initialize();
  import_array();
}

static bool export_type_conversions_once = false;
bool init_and_export_converters() {

  if (export_type_conversions_once)
    return false;
  
  // std::cerr << "PYTHON TYPE CONVERTERS exported" << std::endl;
  export_type_conversions_once = true;

  // Py_Init and array import
  py_init();
  // bot::opencv::export_converters();
  // bot::eigen::export_converters();

  // vectors
  expose_template_type<int>();
  expose_template_type<float>();
  expose_template_type<double>();
  expose_template_type<unsigned long>();

  expose_template_type< std::vector<int> >();
  expose_template_type< std::vector<float> >();
  expose_template_type< std::vector<double> >();
  expose_template_type< std::vector<unsigned long> >();
  
  expose_template_type<int>();
  expose_template_type< std::pair<int, int> >();
  expose_template_type< std::vector<std::pair<int, int> > >();
  
  expose_template_type<std::map<int, std::vector<int> > >();
  expose_template_type<std::map<int, std::vector<float> > >();

  expose_template_type<std::map<std::string, float> >();
  expose_template_type<std::map<std::string, double> >();

  // Expose boost::optional
  // expose_template_type<boost::none>();
  // python_optional<gt::Matrix>();

  return true;
}



BOOST_PYTHON_MODULE(pybot_types)
{
  // Main types export
  bot::python::init_and_export_converters();
  py::scope scope = py::scope();

  // // cv::Point2f
  // py::class_<cv::Point2f>("Point2f")
  //     .def_readwrite("x", &cv::Point2f::x)
  //     .def_readwrite("y", &cv::Point2f::y)
  //     ;
  
  // // cv::KeyPoint
  // py::class_<cv::KeyPoint>("KeyPoint")
  //     .def_readwrite("pt", &cv::KeyPoint::pt)
  //     .def_readwrite("size", &cv::KeyPoint::size)
  //     .def_readwrite("angle", &cv::KeyPoint::angle)
  //     .def_readwrite("response", &cv::KeyPoint::response)
  //     .def_readwrite("octave", &cv::KeyPoint::octave)
  //     .def_readwrite("class_id", &cv::KeyPoint::class_id)
  //     ;
  // expose_template_type< std::vector<cv::KeyPoint> >();
  
}

} // namespace python
} // namespace bot

