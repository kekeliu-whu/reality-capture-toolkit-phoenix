#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Path = std::filesystem::path;

std::uint16_t readLe16(const std::array<std::uint8_t, 18>& bytes, int offset) {
    return static_cast<std::uint16_t>(bytes[offset]) |
           (static_cast<std::uint16_t>(bytes[offset + 1]) << 8U);
}

cv::Mat readRawBgrTga(const Path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open TGA: " + path.string());
    }

    std::array<std::uint8_t, 18> header{};
    input.read(reinterpret_cast<char*>(header.data()), header.size());
    const int width = readLe16(header, 12);
    const int height = readLe16(header, 14);
    if (!input || header[1] != 0 || header[2] != 2 || header[16] != 24 ||
        (header[17] & 0x20U) == 0U || width <= 0 || height <= 0) {
        throw std::runtime_error(
            "Expected an uncompressed top-origin BGR24 TGA: " + path.string());
    }
    input.seekg(header[0], std::ios::cur);

    cv::Mat image(height, width, CV_8UC3);
    input.read(
        reinterpret_cast<char*>(image.data),
        static_cast<std::streamsize>(image.total() * image.elemSize()));
    if (!input) {
        throw std::runtime_error("Truncated TGA payload: " + path.string());
    }
    return image;
}

void writeLe16(std::array<std::uint8_t, 18>& bytes, int offset, int value) {
    bytes[offset] = static_cast<std::uint8_t>(value & 0xff);
    bytes[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xff);
}

void writeRawBgrTga(const Path& path, const cv::Mat& image) {
    if (image.type() != CV_8UC3 || !image.isContinuous() || image.cols > 65535 ||
        image.rows > 65535) {
        throw std::runtime_error("TGA writer requires continuous CV_8UC3 input");
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create TGA: " + path.string());
    }
    std::array<std::uint8_t, 18> header{};
    header[2] = 2;
    writeLe16(header, 12, image.cols);
    writeLe16(header, 14, image.rows);
    header[16] = 24;
    header[17] = 0x20;
    output.write(reinterpret_cast<const char*>(header.data()), header.size());
    output.write(
        reinterpret_cast<const char*>(image.data),
        static_cast<std::streamsize>(image.total() * image.elemSize()));
    if (!output) {
        throw std::runtime_error("Failed while writing TGA: " + path.string());
    }
}

cv::Mat readBinaryPgm(const Path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open PGM: " + path.string());
    }
    std::string magic;
    int width = 0;
    int height = 0;
    int maximum = 0;
    input >> magic >> width >> height >> maximum;
    input.get();
    if (!input || magic != "P5" || width <= 0 || height <= 0 || maximum != 255) {
        throw std::runtime_error("Expected an 8-bit binary PGM: " + path.string());
    }
    cv::Mat mask(height, width, CV_8UC1);
    input.read(
        reinterpret_cast<char*>(mask.data),
        static_cast<std::streamsize>(mask.total()));
    if (!input) {
        throw std::runtime_error("Truncated PGM payload: " + path.string());
    }
    return mask;
}

void writeBinaryPgm(const Path& path, const cv::Mat& mask) {
    if (mask.type() != CV_8UC1 || !mask.isContinuous()) {
        throw std::runtime_error("PGM writer requires continuous CV_8UC1 input");
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create PGM: " + path.string());
    }
    output << "P5\n" << mask.cols << ' ' << mask.rows << "\n255\n";
    output.write(
        reinterpret_cast<const char*>(mask.data),
        static_cast<std::streamsize>(mask.total()));
}

void reduceValid2x2(
    const cv::Mat& source, const cv::Mat& source_mask,
    cv::Mat& reduced, cv::Mat& reduced_mask) {
    const int rows = source.rows / 2;
    const int cols = source.cols / 2;
    reduced = cv::Mat(rows, cols, CV_32FC3, cv::Scalar::all(0));
    reduced_mask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar::all(0));

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            const cv::Vec3f* top = source.ptr<cv::Vec3f>(2 * row);
            const cv::Vec3f* bottom = source.ptr<cv::Vec3f>(2 * row + 1);
            const std::uint8_t* top_mask = source_mask.ptr<std::uint8_t>(2 * row);
            const std::uint8_t* bottom_mask =
                source_mask.ptr<std::uint8_t>(2 * row + 1);
            cv::Vec3f* destination = reduced.ptr<cv::Vec3f>(row);
            std::uint8_t* destination_mask = reduced_mask.ptr<std::uint8_t>(row);

            for (int column = 0; column < cols; ++column) {
                const int source_column = 2 * column;
                cv::Vec3f sum(0.0F, 0.0F, 0.0F);
                int count = 0;
                if (top_mask[source_column] != 0) {
                    sum += top[source_column];
                    ++count;
                }
                if (top_mask[source_column + 1] != 0) {
                    sum += top[source_column + 1];
                    ++count;
                }
                if (bottom_mask[source_column] != 0) {
                    sum += bottom[source_column];
                    ++count;
                }
                if (bottom_mask[source_column + 1] != 0) {
                    sum += bottom[source_column + 1];
                    ++count;
                }
                if (count != 0) {
                    destination[column] =
                        sum * (1.0F / static_cast<float>(count));
                    destination_mask[column] = 255;
                }
            }
        }
    });
}

cv::Mat pyramidInpaintWrapped(const cv::Mat& input, const cv::Mat& input_mask) {
    if (input.type() != CV_8UC3 || input_mask.type() != CV_8UC1 ||
        input.size() != input_mask.size() || (input.cols & 1) != 0) {
        throw std::runtime_error("Floor image/mask types or dimensions are invalid");
    }

    const int horizontal_padding = input.cols / 2;
    cv::Mat padded_image;
    cv::Mat padded_mask;
    cv::copyMakeBorder(
        input, padded_image, 0, 0, horizontal_padding, horizontal_padding,
        cv::BORDER_WRAP);
    cv::copyMakeBorder(
        input_mask, padded_mask, 0, 0, horizontal_padding, horizontal_padding,
        cv::BORDER_WRAP);

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;
    padded_image.convertTo(padded_image, CV_32FC3);
    images.push_back(std::move(padded_image));
    masks.push_back(std::move(padded_mask));

    while (images.back().rows > 1 && images.back().cols > 1) {
        const bool source_fully_valid =
            static_cast<std::size_t>(cv::countNonZero(masks.back())) ==
            masks.back().total();
        cv::Mat reduced;
        cv::Mat reduced_mask;
        reduceValid2x2(images.back(), masks.back(), reduced, reduced_mask);
        if (source_fully_valid) {
            break;
        }
        images.push_back(std::move(reduced));
        masks.push_back(std::move(reduced_mask));
    }

    cv::Mat filled = images.back().clone();
    for (std::size_t level = images.size() - 1; level-- > 0;) {
        cv::Mat expanded;
        cv::resize(
            filled, expanded, images[level].size(), 0.0, 0.0,
            cv::INTER_LINEAR);
        images[level].copyTo(expanded, masks[level]);
        filled = std::move(expanded);
    }

    filled.convertTo(filled, CV_8UC3);
    return filled.colRange(
        horizontal_padding, horizontal_padding + input.cols).clone();
}

cv::Mat buildFloorMask(const std::vector<Path>& projection_mask_paths) {
    if (projection_mask_paths.empty()) {
        throw std::runtime_error("At least one projection mask is required");
    }
    cv::Mat combined = readBinaryPgm(projection_mask_paths.front());
    for (std::size_t index = 1; index < projection_mask_paths.size(); ++index) {
        const cv::Mat next = readBinaryPgm(projection_mask_paths[index]);
        if (next.type() != CV_8UC1 || next.size() != combined.size()) {
            throw std::runtime_error("Projection-mask dimensions do not match");
        }
        cv::bitwise_or(combined, next, combined);
    }
    cv::threshold(combined, combined, 0.0, 1.0, cv::THRESH_BINARY);
    return combined;
}

void requireWrite(bool result, const Path& path) {
    if (!result) {
        throw std::runtime_error("Could not write image: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            std::cerr << "Usage: " << argv[0]
                      << " BLEND_OUTPUT.tga PROJECTION0.pgm PROJECTION1.pgm"
                         " PROJECTION2.pgm PROJECTION3.pgm OUTPUT_DIRECTORY\n";
            return 2;
        }

        const Path blend_path = argv[1];
        const std::vector<Path> projection_masks = {
            argv[2], argv[3], argv[4], argv[5]};
        const Path output_directory = argv[6];
        std::filesystem::create_directories(output_directory);

        const cv::Mat blend_output = readRawBgrTga(blend_path);
        const cv::Mat floor_mask = buildFloorMask(projection_masks);
        if (blend_output.size() != floor_mask.size()) {
            throw std::runtime_error("Blend output and projection masks do not match");
        }

        const Path floor_mask_pgm = output_directory / "floor-mask.pgm";
        const Path binary_mask_png = output_directory / "binary-mask.png";
        const Path no_floor_jpeg = output_directory / "no-floor.jpg";
        const Path floor_input_tga = output_directory / "floor-input.tga";
        const Path floor_output_tga = output_directory / "floor-output.tga";
        const Path filled_jpeg = output_directory / "filled.jpg";

        writeBinaryPgm(floor_mask_pgm, floor_mask);
        requireWrite(
            cv::imwrite(
                binary_mask_png.string(), floor_mask,
                {cv::IMWRITE_PNG_COMPRESSION, 9}),
            binary_mask_png);

        // Empty parameters are effective JPEG quality 95 and optimize=false
        // with the frozen OpenCV ABI.
        requireWrite(cv::imwrite(no_floor_jpeg.string(), blend_output), no_floor_jpeg);
        const cv::Mat floor_input = cv::imread(
            no_floor_jpeg.string(),
            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (floor_input.empty()) {
            throw std::runtime_error("Could not read back no-floor JPEG");
        }
        writeRawBgrTga(floor_input_tga, floor_input);

        const cv::Mat floor_output = pyramidInpaintWrapped(floor_input, floor_mask);
        writeRawBgrTga(floor_output_tga, floor_output);
        requireWrite(
            cv::imwrite(
                filled_jpeg.string(), floor_output,
                {cv::IMWRITE_JPEG_OPTIMIZE, 1}),
            filled_jpeg);

        std::cout << "Wrote exact final-stage candidates to " << output_directory
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
