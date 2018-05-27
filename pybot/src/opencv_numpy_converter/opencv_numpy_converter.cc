// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT
#include <iostream>
#include "stl_numpy_converter/utils/template.h"
#include "stl_numpy_converter/utils/container.h"
#include "opencv_numpy_converter/opencv_numpy_converter.h"

namespace bp = boost::python;

namespace opencv_numpy {

static bool converter_exported = false;
void export_converters() {
  std::cout << "opencv_numpy :: Registering converters (opencv version="
            << CV_VERSION << ")" << std::endl;
  if (converter_exported) {
    std::cout << "opencv_numpy :: Already registered" << std::endl;
    return;
  }

  Py_Initialize();
  // import_array();

  expose_template_type< std::vector<cv::Point> >();
  expose_template_type< std::vector<cv::Point2f> >();
  expose_template_type< std::vector<cv::KeyPoint> >();

  expose_template_type< std::vector<cv::Mat> >();
  expose_template_type< std::vector<cv::Mat1b > >();
  expose_template_type< std::vector<cv::Mat1f > >();

  // std::map => py::dict
  expose_template_type<std::map<int, std::vector<int> > >();
  expose_template_type<std::map<int, std::vector<float> > >();
  expose_template_type<std::map<std::string, float> >();

  // various converters to cv::Mat
  py::to_python_converter<cv::Point, Point_to_mat>();
  py::to_python_converter<cv::Point2f, Point2f_to_mat>();
  py::to_python_converter<cv::Point3f, Point3f_to_mat>();
  py::to_python_converter<cv::Vec3f, Vec3f_to_mat>();

  // register the to-from-python converter for each of the types
  Mat_PyObject_converter< cv::Mat >();

  // 1-channel
  Mat_PyObject_converter< cv::Mat1b >();
  Mat_PyObject_converter< cv::Mat1s >();
  Mat_PyObject_converter< cv::Mat1w >();
  Mat_PyObject_converter< cv::Mat1i >();
  Mat_PyObject_converter< cv::Mat1f >();
  Mat_PyObject_converter< cv::Mat1d >();

  // 2-channel
  Mat_PyObject_converter< cv::Mat2b >();
  Mat_PyObject_converter< cv::Mat2s >();
  Mat_PyObject_converter< cv::Mat2w >();
  Mat_PyObject_converter< cv::Mat2i >();
  Mat_PyObject_converter< cv::Mat2f >();
  Mat_PyObject_converter< cv::Mat2d >();

  // 3-channel
  Mat_PyObject_converter< cv::Mat3b >();
  Mat_PyObject_converter< cv::Mat3s >();
  Mat_PyObject_converter< cv::Mat3w >();
  Mat_PyObject_converter< cv::Mat3i >();
  Mat_PyObject_converter< cv::Mat3f >();
  Mat_PyObject_converter< cv::Mat3d >();

  converter_exported = true;
}

void export_types() {

  // // cv::Point2f
  // py::class_<cv::Point2f>("Point2f")
  //     .def_readwrite("x", &cv::Point2f::x)
  //     .def_readwrite("y", &cv::Point2f::y)
  //     ;
  
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

}  // namespace opencv_numpy
