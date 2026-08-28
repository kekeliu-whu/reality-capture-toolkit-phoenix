#include <ceres/solver.h>

#include <cstddef>
#include <iostream>

#define PRINT_OFFSET(field)                                                \
  std::cout << #field << " " << offsetof(ceres::Solver::Options, field) \
            << "\n"

int main() {
  std::cout << "sizeof " << sizeof(ceres::Solver::Options) << "\n";
  PRINT_OFFSET(max_num_iterations);
  PRINT_OFFSET(num_threads);
  PRINT_OFFSET(initial_trust_region_radius);
  PRINT_OFFSET(trust_region_minimizer_iterations_to_dump);
  PRINT_OFFSET(trust_region_problem_dump_directory);
  PRINT_OFFSET(trust_region_problem_dump_format_type);
  return 0;
}
