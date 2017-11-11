// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include <pybot/opencv_types.hpp>

namespace bot { namespace python {

static void py_init() {
  Py_Initialize();
  import_array();
}

static bool export_cv_type_conversions = false;
bool init_and_export_cv_converters() {

  if (export_cv_type_conversions)
    return false;
  
  // std::cerr << "PYTHON TYPE CONVERTERS exported" << std::endl;
  export_cv_type_conversions = true;

  // Py_Init and array import
  py_init();
  bot::opencv::export_converters();

  return true;
}



BOOST_PYTHON_MODULE(pybot_cv_types)
{
  // Main types export
  py_init();
  pybot::opencv_numpy::export_converters();
  
  py::scope scope = py::scope();

  expose_template_type< std::pair<cv::Mat, cv::Mat> >();

  // cv::Point2f
  py::class_<cv::Point2f>("Point2f")
      .def_readwrite("x", &cv::Point2f::x)
      .def_readwrite("y", &cv::Point2f::y)
      ;
  
  // cv::KeyPoint
  py::class_<cv::KeyPoint>("KeyPoint")
      .def_readwrite("pt", &cv::KeyPoint::pt)
      .def_readwrite("size", &cv::KeyPoint::size)
      .def_readwrite("angle", &cv::KeyPoint::angle)
      .def_readwrite("response", &cv::KeyPoint::response)
      .def_readwrite("octave", &cv::KeyPoint::octave)
      .def_readwrite("class_id", &cv::KeyPoint::class_id)
      ;
  expose_template_type< std::vector<cv::KeyPoint> >();
  
}

} // namespace python
} // namespace bot

