#include <ceres/solver.h>

#include <cstddef>
#include <iostream>

#define PRINT_OFFSET(field) \
    std::cout << #field << ' ' << offsetof(ceres::Solver::Options, field) << '\n'

int main() {
    std::cout << "sizeof " << sizeof(ceres::Solver::Options) << '\n';
    PRINT_OFFSET(minimizer_type);
    PRINT_OFFSET(line_search_direction_type);
    PRINT_OFFSET(line_search_type);
    PRINT_OFFSET(nonlinear_conjugate_gradient_type);
    PRINT_OFFSET(max_lbfgs_rank);
    PRINT_OFFSET(use_approximate_eigenvalue_bfgs_scaling);
    PRINT_OFFSET(line_search_interpolation_type);
    PRINT_OFFSET(min_line_search_step_size);
    PRINT_OFFSET(line_search_sufficient_function_decrease);
    PRINT_OFFSET(max_line_search_step_contraction);
    PRINT_OFFSET(min_line_search_step_contraction);
    PRINT_OFFSET(max_num_line_search_step_size_iterations);
    PRINT_OFFSET(max_num_line_search_direction_restarts);
    PRINT_OFFSET(line_search_sufficient_curvature_decrease);
    PRINT_OFFSET(max_line_search_step_expansion);
    PRINT_OFFSET(trust_region_strategy_type);
    PRINT_OFFSET(dogleg_type);
    PRINT_OFFSET(use_nonmonotonic_steps);
    PRINT_OFFSET(max_consecutive_nonmonotonic_steps);
    PRINT_OFFSET(max_num_iterations);
    PRINT_OFFSET(max_solver_time_in_seconds);
    PRINT_OFFSET(num_threads);
    PRINT_OFFSET(initial_trust_region_radius);
    PRINT_OFFSET(max_trust_region_radius);
    PRINT_OFFSET(min_trust_region_radius);
    PRINT_OFFSET(min_relative_decrease);
    PRINT_OFFSET(min_lm_diagonal);
    PRINT_OFFSET(max_lm_diagonal);
    PRINT_OFFSET(max_num_consecutive_invalid_steps);
    PRINT_OFFSET(function_tolerance);
    PRINT_OFFSET(gradient_tolerance);
    PRINT_OFFSET(parameter_tolerance);
    PRINT_OFFSET(linear_solver_type);
    PRINT_OFFSET(preconditioner_type);
    PRINT_OFFSET(visibility_clustering_type);
    PRINT_OFFSET(residual_blocks_for_subset_preconditioner);
    PRINT_OFFSET(dense_linear_algebra_library_type);
    PRINT_OFFSET(sparse_linear_algebra_library_type);
    PRINT_OFFSET(linear_solver_ordering);
    PRINT_OFFSET(use_explicit_schur_complement);
    PRINT_OFFSET(use_postordering);
    PRINT_OFFSET(dynamic_sparsity);
    PRINT_OFFSET(use_mixed_precision_solves);
    PRINT_OFFSET(max_num_refinement_iterations);
    PRINT_OFFSET(use_inner_iterations);
    PRINT_OFFSET(inner_iteration_ordering);
    PRINT_OFFSET(inner_iteration_tolerance);
    PRINT_OFFSET(min_linear_solver_iterations);
    PRINT_OFFSET(max_linear_solver_iterations);
    PRINT_OFFSET(eta);
    PRINT_OFFSET(jacobi_scaling);
    PRINT_OFFSET(logging_type);
    PRINT_OFFSET(minimizer_progress_to_stdout);
    PRINT_OFFSET(trust_region_minimizer_iterations_to_dump);
    PRINT_OFFSET(trust_region_problem_dump_directory);
    PRINT_OFFSET(trust_region_problem_dump_format_type);
    PRINT_OFFSET(check_gradients);
    PRINT_OFFSET(gradient_check_relative_precision);
    PRINT_OFFSET(gradient_check_numeric_derivative_relative_step_size);
    PRINT_OFFSET(update_state_every_iteration);
    PRINT_OFFSET(callbacks);
}
