// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT
#pragma once

#include <boost/python.hpp>
/*
 * Provides template support
 */

template<typename TemplateType>
struct expose_template_type {
  // do nothing!
};

template<typename TemplateType>
struct expose_template_type_base {
  bool wrapped() {
    namespace bpc = boost::python::converter;
    namespace bp = boost::python;
    bpc::registration const * p = bpc::registry::query(bp::type_id<TemplateType>());
    return p && (p->m_class_object || p->m_to_python);
  }
};
