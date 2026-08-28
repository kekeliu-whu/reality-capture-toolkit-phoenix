#include <ceres/solver.h>

#include <cstddef>
#include <iostream>

#define PRINT_OFFSET(field)                                                   \
  std::cout << #field << " " << offsetof(ceres::Solver::Options, field)     \
            << "\n"

int main() {
  std::cout << "OPTIONS\n";
  std::cout << "sizeof " << sizeof(ceres::Solver::Options) << "\n";
  PRINT_OFFSET(minimizer_type);
  PRINT_OFFSET(trust_region_strategy_type);
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
  PRINT_OFFSET(min_linear_solver_iterations);
  PRINT_OFFSET(max_linear_solver_iterations);
  PRINT_OFFSET(eta);
  PRINT_OFFSET(jacobi_scaling);
  PRINT_OFFSET(logging_type);
  PRINT_OFFSET(minimizer_progress_to_stdout);
  PRINT_OFFSET(update_state_every_iteration);

#undef PRINT_OFFSET
#define PRINT_SUMMARY_OFFSET(field)                                           \
  std::cout << #field << " " << offsetof(ceres::Solver::Summary, field)     \
            << "\n"
  std::cout << "SUMMARY\n";
  std::cout << "sizeof " << sizeof(ceres::Solver::Summary) << "\n";
  PRINT_SUMMARY_OFFSET(iterations);
  PRINT_SUMMARY_OFFSET(initial_cost);
  PRINT_SUMMARY_OFFSET(final_cost);
  PRINT_SUMMARY_OFFSET(num_successful_steps);
  PRINT_SUMMARY_OFFSET(num_unsuccessful_steps);
  PRINT_SUMMARY_OFFSET(termination_type);
  PRINT_SUMMARY_OFFSET(message);
  return 0;
}
