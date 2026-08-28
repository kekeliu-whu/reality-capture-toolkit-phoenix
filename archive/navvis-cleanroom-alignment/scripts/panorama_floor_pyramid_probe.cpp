#include <opencv2/core.hpp>
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

struct TgaHeader {
    std::uint8_t id_length;
    std::uint8_t color_map_type;
    std::uint8_t image_type;
    std::uint16_t color_map_origin;
    std::uint16_t color_map_length;
    std::uint8_t color_map_depth;
    std::uint16_t x_origin;
    std::uint16_t y_origin;
    std::uint16_t width;
    std::uint16_t height;
    std::uint8_t pixel_depth;
    std::uint8_t image_descriptor;
};

std::uint16_t readLe16(const std::array<std::uint8_t, 18>& bytes, int offset) {
    return static_cast<std::uint16_t>(bytes[offset]) |
           (static_cast<std::uint16_t>(bytes[offset + 1]) << 8U);
}

TgaHeader parseTgaHeader(const std::array<std::uint8_t, 18>& bytes) {
    return {
        bytes[0], bytes[1], bytes[2], readLe16(bytes, 3), readLe16(bytes, 5),
        bytes[7], readLe16(bytes, 8), readLe16(bytes, 10), readLe16(bytes, 12),
        readLe16(bytes, 14), bytes[16], bytes[17],
    };
}

cv::Mat readRawBgrTga(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open TGA: " + path.string());
    }

    std::array<std::uint8_t, 18> header_bytes{};
    input.read(reinterpret_cast<char*>(header_bytes.data()), header_bytes.size());
    const TgaHeader header = parseTgaHeader(header_bytes);
    if (!input || header.color_map_type != 0 || header.image_type != 2 ||
        header.pixel_depth != 24 || (header.image_descriptor & 0x20U) == 0U) {
        throw std::runtime_error(
            "Expected an uncompressed, top-origin, 24-bit true-color TGA: " +
            path.string());
    }

    input.seekg(header.id_length, std::ios::cur);
    cv::Mat image(header.height, header.width, CV_8UC3);
    const auto byte_count = static_cast<std::streamsize>(image.total() * image.elemSize());
    input.read(reinterpret_cast<char*>(image.data), byte_count);
    if (!input) {
        throw std::runtime_error("Truncated TGA payload: " + path.string());
    }
    return image;
}

void writeLe16(std::array<std::uint8_t, 18>& bytes, int offset, int value) {
    bytes[offset] = static_cast<std::uint8_t>(value & 0xff);
    bytes[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xff);
}

void writeRawBgrTga(const std::filesystem::path& path, const cv::Mat& image) {
    if (image.type() != CV_8UC3 || image.cols > 65535 || image.rows > 65535) {
        throw std::runtime_error("TGA writer requires a <=65535 CV_8UC3 image");
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
    for (int row = 0; row < image.rows; ++row) {
        output.write(
            reinterpret_cast<const char*>(image.ptr(row)),
            static_cast<std::streamsize>(image.cols * image.elemSize()));
    }
    if (!output) {
        throw std::runtime_error("Failed while writing TGA: " + path.string());
    }
}

cv::Mat readBinaryPgm(const std::filesystem::path& path) {
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

struct PyramidLevelStats {
    int level;
    int width;
    int height;
    std::uint64_t valid;
    bool retained;
};

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

cv::Mat pyramidInpaint(
    const cv::Mat& input, const cv::Mat& input_mask,
    bool horizontal_wrap, std::vector<PyramidLevelStats>& stats) {
    if (input.type() != CV_8UC3 || input_mask.type() != CV_8UC1 ||
        input.size() != input_mask.size()) {
        throw std::runtime_error("Image/mask must be equally sized CV_8UC3/CV_8UC1");
    }

    cv::Mat image = input;
    cv::Mat mask = input_mask;
    int horizontal_padding = 0;
    if (horizontal_wrap) {
        if ((input.cols & 1) != 0) {
            throw std::runtime_error(
                "Horizontal-wrap mode requires an even image width");
        }
        horizontal_padding = input.cols / 2;
        cv::copyMakeBorder(
            input, image, 0, 0, horizontal_padding, horizontal_padding,
            cv::BORDER_WRAP);
        cv::copyMakeBorder(
            input_mask, mask, 0, 0, horizontal_padding, horizontal_padding,
            cv::BORDER_WRAP);
    }

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;
    cv::Mat image_float;
    image.convertTo(image_float, CV_32FC3);
    images.push_back(std::move(image_float));
    masks.push_back(mask);
    stats.push_back({0, image.cols, image.rows,
                     static_cast<std::uint64_t>(cv::countNonZero(mask)), true});

    while (images.back().rows > 1 && images.back().cols > 1) {
        const bool source_fully_valid =
            static_cast<std::size_t>(cv::countNonZero(masks.back())) ==
            masks.back().total();
        cv::Mat reduced;
        cv::Mat reduced_mask;
        reduceValid2x2(images.back(), masks.back(), reduced, reduced_mask);
        const int level = static_cast<int>(stats.size());
        const auto valid =
            static_cast<std::uint64_t>(cv::countNonZero(reduced_mask));
        stats.push_back(
            {level, reduced.cols, reduced.rows, valid, !source_fully_valid});
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

    cv::Mat quantized;
    filled.convertTo(quantized, CV_8UC3);
    if (!horizontal_wrap) {
        return quantized;
    }
    return quantized.colRange(
        horizontal_padding, horizontal_padding + input.cols).clone();
}

void writeStats(
    const std::filesystem::path& path,
    const std::vector<PyramidLevelStats>& stats,
    bool horizontal_wrap) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create stats JSON: " + path.string());
    }
    output << "{\n  \"horizontal_wrap\": "
           << (horizontal_wrap ? "true" : "false") << ",\n  \"levels\": [\n";
    for (std::size_t index = 0; index < stats.size(); ++index) {
        const auto& level = stats[index];
        const std::uint64_t total =
            static_cast<std::uint64_t>(level.width) * level.height;
        output << "    {\"level\": " << level.level
               << ", \"width\": " << level.width
               << ", \"height\": " << level.height
               << ", \"valid\": " << level.valid
               << ", \"total\": " << total
               << ", \"retained\": "
               << (level.retained ? "true" : "false") << "}";
        output << (index + 1 == stats.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6) {
            std::cerr
                << "Usage: " << argv[0]
                << " INPUT.tga MASK.pgm OUTPUT.tga [STATS.json] [--no-wrap]\n";
            return 2;
        }
        const std::filesystem::path input_path = argv[1];
        const std::filesystem::path mask_path = argv[2];
        const std::filesystem::path output_path = argv[3];
        std::filesystem::path stats_path;
        bool horizontal_wrap = true;
        for (int index = 4; index < argc; ++index) {
            if (std::string(argv[index]) == "--no-wrap") {
                horizontal_wrap = false;
            } else {
                stats_path = argv[index];
            }
        }

        const cv::Mat image = readRawBgrTga(input_path);
        const cv::Mat mask = readBinaryPgm(mask_path);
        std::vector<PyramidLevelStats> stats;
        const cv::Mat result = pyramidInpaint(image, mask, horizontal_wrap, stats);
        writeRawBgrTga(output_path, result);
        if (!stats_path.empty()) {
            writeStats(stats_path, stats, horizontal_wrap);
        }
        std::cout << "wrote " << output_path << "; levels=" << stats.size()
                  << "; wrap=" << (horizontal_wrap ? "true" : "false")
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
