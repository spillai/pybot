cmake_minimum_required(VERSION 2.8)

project(PYBOT C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_MODULE_PATH ${PYBOT_SOURCE_DIR}/cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")

# Check the architecture and set CMAKE_LIBRARY_ARCHITECTURE accordingly
if(UNIX)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
  endif()
endif()

# cmake Dependencies
include(cmake/FindNumPy.cmake)
include(cmake/FindEigen3.cmake)
include(cmake/FindCholmod.cmake)
# include(cmake/FindGlog.cmake)
include(cmake/boost-python.cmake)

# find_package(Ceres REQUIRED)
# include_directories(${CERES_INCLUDE_DIR})

find_package(Gflags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


set(BOOST_NUMPY_LIBS
   ${GFLAGS_LIBRARY} ${GLOG_LIBRARY}
   ${Boost_LIBRARIES}
   # boost_system boost_python glog
   ${CHOLMOD_LIBRARIES}
   ${PYTHON_LIBRARIES})

# Eigen-NumPy converter
add_library(eigen_numpy SHARED eigen_numpy_conversion.cpp)
target_link_libraries(eigen_numpy ${BOOST_NUMPY_LIBS})
set_target_properties(eigen_types PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${PYBOT_SOURCE_DIR}/.."
)

# Opencv-NumPy converter
add_library(opencv_numpy SHARED opencv_numpy_conversion.cpp)
target_link_libraries(opencv_numpy ${BOOST_NUMPY_LIBS})
set_target_properties(opencv_types PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${PYBOT_SOURCE_DIR}/.."
)


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

