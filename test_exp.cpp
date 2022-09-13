#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <exception>
#include <iostream>
using namespace boost::python;
struct my_exception : std::exception
{
  char const* what() const throw() { return "One of my exceptions"; }
};

void translate(my_exception const& e)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

void something_which_throws()
{
    //...
    std::cout << "test" << std::endl;
    throw my_exception();
    //...
}
void test()
{

    std::cout << "test1" << std::endl;
}

BOOST_PYTHON_MODULE(test_exp)
{
  register_exception_translator<my_exception>(&translate);
  def("something_which_throws", something_which_throws);
  def("test", test);
}