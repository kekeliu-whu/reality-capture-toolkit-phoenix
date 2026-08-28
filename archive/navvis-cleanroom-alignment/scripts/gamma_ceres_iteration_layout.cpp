#include <ceres/iteration_callback.h>

#include <cstddef>
#include <iostream>

#define PRINT_OFFSET(field)                                                   \
  std::cout << #field << " " << offsetof(ceres::IterationSummary, field)    \
            << "\n"

int main() {
  std::cout << "sizeof " << sizeof(ceres::IterationSummary) << "\n";
  PRINT_OFFSET(iteration);
  PRINT_OFFSET(step_is_valid);
  PRINT_OFFSET(step_is_nonmonotonic);
  PRINT_OFFSET(step_is_successful);
  PRINT_OFFSET(cost);
  PRINT_OFFSET(cost_change);
  PRINT_OFFSET(gradient_max_norm);
  PRINT_OFFSET(gradient_norm);
  PRINT_OFFSET(step_norm);
  PRINT_OFFSET(relative_decrease);
  PRINT_OFFSET(trust_region_radius);
  PRINT_OFFSET(eta);
  PRINT_OFFSET(step_size);
  PRINT_OFFSET(linear_solver_iterations);
  PRINT_OFFSET(iteration_time_in_seconds);
  PRINT_OFFSET(step_solver_time_in_seconds);
  PRINT_OFFSET(cumulative_time_in_seconds);
}
