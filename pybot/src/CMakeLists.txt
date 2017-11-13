cmake_minimum_required(VERSION 2.8)

project(PYBOT C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_MODULE_PATH ${PYBOT_SOURCE_DIR}/cmake)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -fpermissive")

# Check the architecture and set CMAKE_LIBRARY_ARCHITECTURE accordingly
if(UNIX)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
  endif()
endif()

# cmake Dependencies
include(cmake/FindNumPy.cmake)
include(cmake/FindEigen3.cmake)
include(cmake/FindGlog.cmake)
include(cmake/boost-python.cmake)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIR})

find_package(Gflags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${NUMPY_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

set(BOOST_NUMPY_LIBS
   ${GFLAGS_LIBRARY} ${GLOG_LIBRARY}
   ${Boost_LIBRARIES}
   ${PYTHON_LIBRARIES})

# STL-NumPy converter
boost_python_module(stl_numpy
                    stl_numpy_converter/stl_numpy_converter.cc)
target_link_libraries(stl_numpy ${BOOST_NUMPY_LIBS})

# Eigen-NumPy converter
boost_python_module(eigen_numpy
                    eigen_numpy_converter/eigen_numpy_converter.cc)
target_link_libraries(eigen_numpy ${BOOST_NUMPY_LIBS} stl_numpy)

# Opencv-NumPy converter
boost_python_module(opencv_numpy
                    opencv_numpy_converter/conversion.cc
                    opencv_numpy_converter/opencv_numpy_converter.cc)
target_link_libraries(opencv_numpy ${BOOST_NUMPY_LIBS} ${OpenCV_LIBS} stl_numpy)

# pybot-types
boost_python_module(pybot_types pybot_types.cpp)
target_link_libraries(pybot_types ${BOOST_NUMPY_LIBS} ${OpenCV_LIBS}
                      stl_numpy eigen_numpy opencv_numpy)
set_target_properties(pybot_types PROPERTIES
                      PREFIX ""
                      SUFFIX ".so"
                      LIBRARY_OUTPUT_DIRECTORY "${PYBOT_SOURCE_DIR}/..")

# # Python wrapper
# add_library(eigen_types SHARED eigen_types.cpp)
# target_link_libraries(eigen_types
#     ${OpenCV_LIBS}
#     ${GFLAGS_LIBRARY}
#     ${GLOG_LIBRARY}
#     # ${CERES_LIBRARIES}
#     # ${LAPACK_LIBRARIES}
#     # ${SUITESPARSE_LIBRARIES}
#     ${Boost_LIBRARIES}
# )
# set_target_properties(eigen_types PROPERTIES
#     PREFIX ""
#     SUFFIX ".so"
#     LIBRARY_OUTPUT_DIRECTORY "${PYBOT_SOURCE_DIR}/.."
# )
# if(APPLE)
#     set_target_properties(eigen_types PROPERTIES
#         LINK_FLAGS "-undefined dynamic_lookup"
#     )
# else()
#     target_link_libraries(eigen_types
#         ${PYTHON_LIBRARIES}
#     )
# endif()


# # =========== pybot_eigen_types Python wrapper ==============
# boost_python_module(pybot_eigen_types
#     src/eigen_types.cpp
#     src/eigen_numpy_conversion.cpp)
# target_link_libraries(pybot_eigen_types boost_system boost_python glog ${CHOLMOD_LIBRARIES})
# install(TARGETS pybot_eigen_types
#   PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib
# )

