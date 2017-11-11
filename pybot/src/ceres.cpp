#include <iostream>
#include <ceres/ceres.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <glog/logging.h>

using namespace pybind11::literals;
namespace py = pybind11;

class CustomFunction : public ceres::FirstOrderFunction {
    public:
        CustomFunction(py::function func,
                       py::function grad,
                       unsigned long num_params)
                        : function_(func),
                          gradient_(grad),
                          num_params_(num_params) {}
        virtual ~CustomFunction() override {}
        virtual bool Evaluate(const double* parameters,
                              double* cost,
                              double* gradient) const override {
            auto arr = py::array(py::buffer_info(
                const_cast<double*>(parameters),
                sizeof(double),
                py::format_descriptor<double>::value,
                1,
                { num_params_ },
                { sizeof(double) }
            ));

            auto out = function_(arr);

            // If the cost evaluation fails, the function should return None.
            // Returning false means that Ceres should try other parameters.
            if (out == py::none()) {
                return false;
            }

            cost[0] = out.cast<double>();

            // If gradient is not null, we are supposed to
            // calculate the gradient.
            if (gradient) {
                auto out = gradient_(arr);

                // We also check the gradient for None.
                if (out == py::none()) {
                    return false;
                }

                auto arr = out.cast<py::array_t<double>>();
                auto buf = static_cast<double*>(arr.request().ptr);
                for (unsigned long i = 0; i < num_params_; i++) {
                    gradient[i] = buf[i];
                }
            }
            return true;
        }
        virtual int NumParameters() const override {
            return num_params_;
        }
    private:
        py::function function_;
        py::function gradient_;
        unsigned long num_params_;
};

py::object optimize(py::function func, py::function grad, py::buffer x0) {
    auto buf1 = x0.request();
    if (buf1.ndim != 1) {
        throw std::runtime_error("Number of dimensions of x0 must be one");
    }
    ceres::GradientProblemSolver::Options options;
    //options.minimizer_progress_to_stdout = true;
    ceres::GradientProblemSolver::Summary summary;
    ceres::GradientProblem problem(new CustomFunction(func, grad, buf1.size));

    auto result = py::array_t<double>(buf1.size);
    auto buf2 = result.request();

    double* data = static_cast<double*>(buf2.ptr);
    double* inputs = static_cast<double*>(buf1.ptr);

    for (unsigned long i = 0; i < buf1.size; i++) {
        data[i] = inputs[i];
    }

    ceres::Solve(options, problem, data, &summary);

    auto iterations = summary.iterations;
    int nfev = 0;
    int ngev = 0;
    for (auto& summ: iterations) {
        nfev += summ.line_search_function_evaluations;
        ngev += summ.line_search_gradient_evaluations;
    }

    auto OptimizeResult = py::module::import("scipy.optimize").attr("OptimizeResult");

    py::dict out("x"_a = result,
                 "success"_a = summary.termination_type == ceres::CONVERGENCE ||
                               summary.termination_type == ceres::USER_SUCCESS,
                 "status"_a = (int)summary.termination_type,
                 "message"_a = summary.message,
                 "fun"_a = summary.final_cost,
                 "nfev"_a = nfev,
                 "ngev"_a = ngev);

    return OptimizeResult(out);
}
