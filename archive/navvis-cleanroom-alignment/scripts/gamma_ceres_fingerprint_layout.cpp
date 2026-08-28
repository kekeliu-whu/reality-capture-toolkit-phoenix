#include <ceres/iteration_callback.h>
#include <ceres/solver.h>

#include <cstddef>
#include <iostream>

#define PRINT_OFFSET(type, field) \
    std::cout << #type "." #field "=" << offsetof(type, field) << '\n'

int main() {
    std::cout << "Solver::Options.size=" << sizeof(ceres::Solver::Options) << '\n';
    PRINT_OFFSET(ceres::Solver::Options, min_linear_solver_iterations);
    PRINT_OFFSET(ceres::Solver::Options, max_linear_solver_iterations);
    PRINT_OFFSET(ceres::Solver::Options, eta);
    PRINT_OFFSET(ceres::Solver::Options, jacobi_scaling);
    PRINT_OFFSET(ceres::Solver::Options, logging_type);
    PRINT_OFFSET(ceres::Solver::Options, minimizer_progress_to_stdout);

    std::cout << "Solver::Summary.size=" << sizeof(ceres::Solver::Summary) << '\n';
    PRINT_OFFSET(ceres::Solver::Summary, iterations);
    PRINT_OFFSET(ceres::Solver::Summary, initial_cost);
    PRINT_OFFSET(ceres::Solver::Summary, final_cost);
    PRINT_OFFSET(ceres::Solver::Summary, num_successful_steps);
    PRINT_OFFSET(ceres::Solver::Summary, num_unsuccessful_steps);

    std::cout << "IterationSummary.size=" << sizeof(ceres::IterationSummary) << '\n';
    PRINT_OFFSET(ceres::IterationSummary, iteration);
    PRINT_OFFSET(ceres::IterationSummary, step_is_valid);
    PRINT_OFFSET(ceres::IterationSummary, step_is_nonmonotonic);
    PRINT_OFFSET(ceres::IterationSummary, step_is_successful);
    PRINT_OFFSET(ceres::IterationSummary, cost);
    PRINT_OFFSET(ceres::IterationSummary, cost_change);
    PRINT_OFFSET(ceres::IterationSummary, gradient_max_norm);
    PRINT_OFFSET(ceres::IterationSummary, gradient_norm);
    PRINT_OFFSET(ceres::IterationSummary, step_norm);
    PRINT_OFFSET(ceres::IterationSummary, relative_decrease);
    PRINT_OFFSET(ceres::IterationSummary, trust_region_radius);
    PRINT_OFFSET(ceres::IterationSummary, eta);
    PRINT_OFFSET(ceres::IterationSummary, linear_solver_iterations);
    PRINT_OFFSET(ceres::IterationSummary, iteration_time_in_seconds);
    PRINT_OFFSET(ceres::IterationSummary, step_solver_time_in_seconds);
    PRINT_OFFSET(ceres::IterationSummary, cumulative_time_in_seconds);
}
