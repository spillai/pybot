// Copyright(c) 2008 Tri Tech Information Systems Inc. 
// Distributed under the Boost Software License, Version 1.0.
//     (See accompanying file ../../LICENSE_1_0.txt or copy at
//           http://www.boost.org/LICENSE_1_0.txt)
//     

// Modified by: Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include <boost/python.hpp>
#include <utility>

// This templates make it so that any pair will become a tuple

template<typename A, typename B>
PyObject * pair_to_tuple(std::pair<A, B> input)
{
    using namespace boost::python;
    return incref( boost::python::make_tuple(input.first, input.second).ptr() );
}

namespace boost
{
    namespace python
    {
        template <typename A, typename B>
        struct to_python_value< std::pair<A,B>& > : detail::builtin_to_python
        {
            inline PyObject* operator ()(std::pair<A,B> const & x) const
            {
                return pair_to_tuple(x);
            }

            static PyTypeObject const *get_pytype()
            {   return PyTuple_Type;
            }
        };

        template <typename A, typename B>
        struct to_python_value< std::pair<A,B> const & > : detail::builtin_to_python
        {
            inline PyObject* operator ()(std::pair<A,B> const & x) const
            {
                return pair_to_tuple(x);
            }

            static PyTypeObject const *get_pytype()
            {   return &PyTuple_Type;
            }
        };


        namespace converter
        {
            template <typename A, typename B>
            struct arg_to_python< std::pair<A,B> > : handle<>
            {
                arg_to_python( std::pair<A,B> const & x) : python::handle<>( pair_to_tuple(x) )
                {}
            };            
        }
    }
}
;
