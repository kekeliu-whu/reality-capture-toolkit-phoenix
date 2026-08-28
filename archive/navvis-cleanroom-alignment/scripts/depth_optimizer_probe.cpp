#include "navvis_recon/panorama_rendering.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename Value>
void readRows(const std::string& path, cv::Mat_<Value>& matrix) {
    std::ifstream input(path, std::ios::binary);
    for (int row = 0; row < matrix.rows; ++row) {
        input.read(
            reinterpret_cast<char*>(matrix.template ptr<Value>(row)),
            static_cast<std::streamsize>(matrix.cols * sizeof(Value)));
    }
    if (!input) {
        throw std::runtime_error("Cannot read matrix: " + path);
    }
}

template <typename Value>
void writeRows(const std::string& path, const cv::Mat_<Value>& matrix) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (int row = 0; row < matrix.rows; ++row) {
        output.write(
            reinterpret_cast<const char*>(matrix.template ptr<Value>(row)),
            static_cast<std::streamsize>(matrix.cols * sizeof(Value)));
    }
    if (!output) {
        throw std::runtime_error("Cannot write matrix: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            std::cerr << "Usage: depth_optimizer_probe ROWS COLS MEASURED_F64 MASK_U8 OUTPUT_F64\n";
            return 2;
        }
        const int rows = std::stoi(argv[1]);
        const int columns = std::stoi(argv[2]);
        cv::Mat1d measured_double(rows, columns);
        cv::Mat1b mask(rows, columns);
        readRows(argv[3], measured_double);
        readRows(argv[4], mask);

        cv::Mat1f measured;
        measured_double.convertTo(measured, CV_32F);
        const cv::Mat1d optimized =
            navvis_recon::GaussNewtonDepthMapOptimizer::optimizeDouble(
                measured, mask, 1.0, 10000);
        writeRows(argv[5], optimized);
        std::cout << "Wrote " << columns << 'x' << rows << " optimized depth\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
