// The following code will make it possible to use boost::optional<T>
// very easily with Boost.Python.  For every time you expect to be used
// optionally, you must call the following during module initialization:

//     python_optional<some_type_t>();

// You can now accept and return boost::optional<some_type_t>().  In
//     both directions, None will reflect an uninitialized value, while data
//     of the correct type represents an initialized value.  Needless to
//     say, valid conversions must already exist for some_type_t, otherwise
//     an error is generated claiming that the optional type cannot be
//     converted.
// 
// https://mail.python.org/pipermail/cplusplus-sig/2007-May/012003.html

// Modified by: Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#ifndef UTILS_OPTIONAL_H_
#define UTILS_OPTIONAL_H_

#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include <boost/none.hpp>

template <typename T, typename TfromPy>
    struct object_from_python
    {
      object_from_python() {
        boost::python::converter::registry::push_back
            (&TfromPy::convertible, &TfromPy::construct,
             boost::python::type_id<T>());
      }
    };

template <typename T, typename TtoPy, typename TfromPy>
    struct register_python_conversion
    {
      register_python_conversion() {
        boost::python::to_python_converter<T, TtoPy>();
        object_from_python<T, TfromPy>();
      }
    };

template <typename T>
struct python_optional : public boost::noncopyable
{
  struct optional_to_python
  {
    static PyObject * convert(const boost::optional<T>& value)
    {
      return (value ? boost::python::to_python_value<T>()(*value) :
              boost::python::detail::none());
    }
  };

  struct optional_from_python
  {
    static void * convertible(PyObject * source)
    {
      using namespace boost::python::converter;

      if (source == Py_None)
        return source;

      const registration& converters(registered<T>::converters);

      if (implicit_rvalue_convertible_from_python(source,
                                                  converters)) {
        rvalue_from_python_stage1_data data =
            rvalue_from_python_stage1(source, converters);
        return rvalue_from_python_stage2(source, data, converters);
      }
      return NULL;
    }

    static void construct(PyObject * source,
                          boost::python::converter::rvalue_from_python_stage1_data * data)
    {
      using namespace boost::python::converter;

      void * const storage = ((rvalue_from_python_storage<T> *)
                              data)->storage.bytes;

      if (data->convertible == source)    // == None
        new (storage) boost::optional<T>(); // A Boost uninitialized value
      else
        new (storage) boost::optional<T>(*static_cast<T *>(data->convertible));

      data->convertible = storage;
    }
  };

  explicit python_optional() {
    register_python_conversion<boost::optional<T>,
        optional_to_python, optional_from_python>();
  }
};


#endif // UTILS_OPTIONAL_H_
