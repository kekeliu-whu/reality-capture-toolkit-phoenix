#include "navvis_recon/mapped_space_quality.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DiskRecordV2 {
    std::int32_t key_x;
    std::int32_t key_y;
    std::int32_t key_z;
    float xyz_x;
    float xyz_y;
    float xyz_z;
    float origin_x;
    float origin_y;
    float origin_z;
    float normal_x;
    float normal_y;
    float normal_z;
    float intensity;
    std::uint32_t count;
};
static_assert(sizeof(DiskRecordV2) == 56U);

struct Options {
    fs::path input_shards;
    fs::path output_directory;
    navvis_recon::MappedSpaceQualityOptions quality;
};

std::string requireValue(int& index, const int argc, char** argv, const std::string& name) {
    if (++index >= argc) {
        throw std::invalid_argument("missing value after " + name);
    }
    return argv[index];
}

Options parseOptions(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--input-shards") {
            options.input_shards = requireValue(index, argc, argv, argument);
        } else if (argument == "--output-dir") {
            options.output_directory = requireValue(index, argc, argv, argument);
        } else if (argument == "--grid-resolution") {
            options.quality.voxel_size = std::stod(requireValue(index, argc, argv, argument));
        } else if (argument == "--min-num-rays-per-voxel") {
            options.quality.minimum_rays_per_voxel =
                std::stoi(requireValue(index, argc, argv, argument));
        } else if (argument == "--use-every-nth-point") {
            options.quality.use_every_nth_point =
                std::stoi(requireValue(index, argc, argv, argument));
        } else if (argument == "--brotli-quality") {
            options.quality.brotli_quality =
                std::stoi(requireValue(index, argc, argv, argument));
        } else if (argument == "--max-ray-length") {
            // This option controls the original estimator's spatial tiling,
            // not clipping. The shard implementation does not need it.
            static_cast<void>(std::stod(requireValue(index, argc, argv, argument)));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: navvis_recon_mapped_space_quality --input-shards DIR --output-dir DIR\n"
                << "  [--grid-resolution 0.16666666666666669]\n"
                << "  [--min-num-rays-per-voxel 36] [--use-every-nth-point 1]\n"
                << "  [--brotli-quality 5] [--max-ray-length 50]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown mapped-space quality option: " + argument);
        }
    }
    if (options.input_shards.empty() || options.output_directory.empty()) {
        throw std::invalid_argument("--input-shards and --output-dir are required");
    }
    return options;
}

std::vector<fs::path> inputFiles(const fs::path& directory) {
    if (!fs::is_directory(directory)) {
        throw std::runtime_error("mapped-space quality input directory is missing: " + directory.string());
    }
    std::vector<fs::path> files;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".raytile") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("mapped-space quality found no .raytile input files");
    }
    return files;
}

std::uint64_t addShard(const fs::path& path,
                       navvis_recon::MappedSpaceQualityGrid& grid) {
    const std::uintmax_t bytes = fs::file_size(path);
    if (bytes == 0U || bytes % sizeof(DiskRecordV2) != 0U) {
        throw std::runtime_error("invalid or partial .raytile file: " + path.string());
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }

    std::uint64_t records_read = 0U;
    DiskRecordV2 record{};
    while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
        if (record.count != 1U) {
            throw std::runtime_error(
                "mapped-space quality requires exact per-ray V2 shards; "
                "clustered count != 1 in " +
                path.string());
        }

        navvis_recon::MappedSpaceQualityRay ray;
        ray.origin = Eigen::Vector3d(record.origin_x, record.origin_y, record.origin_z);
        ray.endpoint = Eigen::Vector3d(record.xyz_x, record.xyz_y, record.xyz_z);
        grid.addRay(ray);
        ++records_read;
    }
    if (!input.eof()) {
        throw std::runtime_error("read failure in " + path.string());
    }
    return records_read;
}

std::uint64_t addShards(const std::vector<fs::path>& files,
                        navvis_recon::MappedSpaceQualityGrid& grid) {
    std::uint64_t records_read = 0U;
    for (const fs::path& path : files) {
        records_read += addShard(path, grid);
    }
    return records_read;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        navvis_recon::MappedSpaceQualityGrid grid(options.quality);
        const std::vector<fs::path> files = inputFiles(options.input_shards);
        const std::uint64_t records_read = addShards(files, grid);
        const std::vector<navvis_recon::CompactQualityVoxel> voxels = grid.compact();
        navvis_recon::writeMappedSpaceQuality(options.output_directory, voxels, options.quality);
        std::cout << "mapped-space quality: " << records_read << " rays -> " << voxels.size()
                  << " voxels in " << options.output_directory << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mapped-space quality failed: " << error.what() << '\n';
        return 1;
    }
}
