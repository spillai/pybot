// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#pragma once

#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/foreach.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <map>
#include <list>

namespace py = boost::python;

template<typename containedType>
struct custom_list_to_list{
  static PyObject* convert(const std::list<containedType>& v){
    py::list ret; BOOST_FOREACH(const containedType& e, v) ret.append(e);
    return py::incref(ret.ptr());
  }
};

template<typename T>
struct expose_template_type< std::vector<T> > :
    public expose_template_type_base< std::vector<T> > {
  typedef expose_template_type_base< std::vector<T> > base_type;
  typedef expose_template_type< std::vector<T> > this_type;
  typedef std::vector<T> wrapped_type;

  expose_template_type() {
    ::expose_template_type<T>();
    if (!base_type::wrapped()) {
      boost::python::to_python_converter< wrapped_type, this_type >();
      boost::python::converter::registry::push_back(
          this_type::convertible,
          this_type::construct,
          boost::python::type_id< wrapped_type >() );
    }
  }

  static PyObject* convert(const wrapped_type & container) {
    boost::python::list l;
    for (typename wrapped_type::const_iterator iter = container.begin(); iter != container.end(); iter++) {
      l.append(boost::python::object(*iter));
    }
    Py_INCREF(l.ptr());
    return l.ptr();
  }

  static void* convertible(PyObject* py_obj) {
    // we are supposed to indicate whether or not we can convert this
    // we don't really know, but we'll try any sequence
    if (PySequence_Check(py_obj))
      return py_obj;
    return 0;
  }

  static void construct(PyObject* py_obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
    namespace bp = boost::python;
    typedef bp::converter::rvalue_from_python_storage< wrapped_type > storage_t;

    storage_t* the_storage = reinterpret_cast<storage_t*>(data);
    void* memory_chunk = the_storage->storage.bytes;
    wrapped_type * newvec = new (memory_chunk) wrapped_type;
    data->convertible = memory_chunk;

    bp::object sequence(bp::handle<>(bp::borrowed(py_obj)));

    for (int idx = 0; idx < len(sequence); idx++) {
      newvec->push_back(bp::extract<T>(sequence[idx])());
    }
  }
};

template<typename Key, typename Value>
struct expose_template_type< std::map<Key, Value> > :
    public expose_template_type_base< std::map<Key, Value> > {
  typedef std::map<Key, Value> wrapped_type;
  typedef expose_template_type_base< wrapped_type > base_type;
  typedef expose_template_type< wrapped_type > this_type;

  expose_template_type() {
    if (!base_type::wrapped()) {
      boost::python::to_python_converter< wrapped_type, this_type >();
      boost::python::converter::registry::push_back(
          this_type::convertible,
          this_type::construct,
          boost::python::type_id< wrapped_type >() );
    }
  }

  static PyObject* convert(const wrapped_type& container) {
    boost::python::dict d;
    for (typename wrapped_type::const_iterator iter = container.begin(); iter != container.end(); iter++) {
      d[iter->first] = boost::python::object(iter->second);
    }
    Py_INCREF(d.ptr());
    return d.ptr();
  }

  static void* convertible(PyObject* py_obj) {
    // we are supposed to indicate whether or not we can convert this
    // we don't really know, but we'll try any sequence
    if (PyMapping_Check(py_obj))
      return py_obj;
    return 0;
  }

  static void construct(PyObject* py_obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
    namespace bp = boost::python;
    typedef bp::converter::rvalue_from_python_storage< wrapped_type > storage_t;

    storage_t* the_storage = reinterpret_cast<storage_t*>(data);
    void* memory_chunk = the_storage->storage.bytes;
    wrapped_type * newvec = new (memory_chunk) wrapped_type;
    data->convertible = memory_chunk;

    bp::object sequence(bp::handle<>(bp::borrowed(py_obj)));
    sequence = sequence.attr("items")();

    for (int idx = 0; idx < len(sequence); idx++) {
      Key key = bp::extract<Key>(sequence[idx][0])();
      Value value = bp::extract<Value>(sequence[idx][1])();
      (*newvec)[key] = value;
    }
  }
};

template<typename Key, typename Value>
struct expose_template_type< typename std::pair<Key, Value> > :
    public expose_template_type_base< std::pair<Key, Value> > {
    typedef std::pair<Key, Value> wrapped_type;
    typedef expose_template_type_base< wrapped_type > base_type;
    typedef expose_template_type< wrapped_type > this_type;

  expose_template_type() {
    if (!base_type::wrapped()) {
      boost::python::converter::registry::push_back(
          this_type::convertible,
          this_type::construct,
          boost::python::type_id< wrapped_type >() );
    }
  }

  static void* convertible(PyObject* py_obj) {
    // we are supposed to indicate whether or not we can convert this
    // we don't really know, but we'll try any sequence
    if (PyTuple_Check(py_obj) && PyTuple_Size(py_obj) == 2)
      return py_obj;
    return 0;
  }

  static void construct(PyObject * py_obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
    namespace bp = boost::python;
    typedef bp::converter::rvalue_from_python_storage< wrapped_type > storage_t;

    storage_t* the_storage = reinterpret_cast<storage_t*>(data);
    void* memory_chunk = the_storage->storage.bytes;
    wrapped_type * newvec = new (memory_chunk) wrapped_type;
    data->convertible = memory_chunk;

    bp::object sequence(bp::handle<>(bp::borrowed(py_obj)));
    newvec->first = bp::extract<Key>(sequence[0])();
    newvec->second = bp::extract<Value>(sequence[1])();
  }
};
