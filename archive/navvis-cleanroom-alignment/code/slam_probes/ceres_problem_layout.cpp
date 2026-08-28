#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define private public
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#undef private

#define PRINT_OFFSET(type, field) \
    std::cout << #type "." #field << " " << offsetof(type, field) << '\n'

int main() {
    using ceres::internal::ProblemImpl;
    using ceres::internal::Program;
    using ceres::internal::ResidualBlock;

    std::cout << "ProblemImpl.size " << sizeof(ProblemImpl) << '\n';
    PRINT_OFFSET(ProblemImpl, options_);
    PRINT_OFFSET(ProblemImpl, context_impl_owned_);
    PRINT_OFFSET(ProblemImpl, context_impl_);
    PRINT_OFFSET(ProblemImpl, parameter_block_map_);
    PRINT_OFFSET(ProblemImpl, residual_block_set_);
    PRINT_OFFSET(ProblemImpl, program_);

    std::cout << "Program.size " << sizeof(Program) << '\n';
    PRINT_OFFSET(Program, parameter_blocks_);
    PRINT_OFFSET(Program, residual_blocks_);
    PRINT_OFFSET(Program, evaluation_callback_);

    std::cout << "ResidualBlock.size " << sizeof(ResidualBlock) << '\n';
    PRINT_OFFSET(ResidualBlock, cost_function_);
    PRINT_OFFSET(ResidualBlock, loss_function_);
    PRINT_OFFSET(ResidualBlock, parameter_blocks_);
    PRINT_OFFSET(ResidualBlock, index_);
    return 0;
}
