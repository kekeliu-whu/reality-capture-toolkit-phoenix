#include "navvis_recon/mapped_space_quality.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    int threads = 0;
    std::uint64_t chunk_rays = 4'000'000U;
};

struct InputFile {
    fs::path path;
    std::uint64_t records = 0U;
    std::uint64_t global_start = 0U;
};

struct WorkItem {
    std::uint64_t global_start = 0U;
    std::uint64_t records = 0U;
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
        } else if (argument == "--threads") {
            options.threads = std::stoi(requireValue(index, argc, argv, argument));
        } else if (argument == "--chunk-rays") {
            options.chunk_rays =
                std::stoull(requireValue(index, argc, argv, argument));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: navvis_recon_mapped_space_quality --input-shards DIR --output-dir DIR\n"
                << "  [--grid-resolution 0.16666666666666669]\n"
                << "  [--min-num-rays-per-voxel 36] [--use-every-nth-point 1]\n"
                << "  [--brotli-quality 5] [--max-ray-length 50]\n"
                << "  [--threads 0=auto] [--chunk-rays 4000000]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown mapped-space quality option: " + argument);
        }
    }
    if (options.input_shards.empty() || options.output_directory.empty()) {
        throw std::invalid_argument("--input-shards and --output-dir are required");
    }
    if (options.threads < 0 || options.chunk_rays == 0U) {
        throw std::invalid_argument("--threads must be non-negative and --chunk-rays positive");
    }
    return options;
}

std::vector<InputFile> inputFiles(const fs::path& directory) {
    if (!fs::is_directory(directory)) {
        throw std::runtime_error("mapped-space quality input directory is missing: " + directory.string());
    }
    std::vector<fs::path> paths;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".raytile") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error("mapped-space quality found no .raytile input files");
    }
    std::vector<InputFile> files;
    files.reserve(paths.size());
    std::uint64_t global_start = 0U;
    for (const fs::path& path : paths) {
        const std::uintmax_t bytes = fs::file_size(path);
        if (bytes == 0U || bytes % sizeof(DiskRecordV2) != 0U) {
            throw std::runtime_error("invalid or partial .raytile file: " + path.string());
        }
        const std::uint64_t records = bytes / sizeof(DiskRecordV2);
        files.push_back({path, records, global_start});
        global_start += records;
    }
    return files;
}

std::uint64_t totalRecords(const std::vector<InputFile>& files) {
    std::uint64_t total = 0U;
    for (const InputFile& file : files) {
        total += file.records;
    }
    return total;
}

std::vector<WorkItem> workItems(
    const std::uint64_t total_records, const std::uint64_t requested_chunk_rays,
    const int threads) {
    const std::uint64_t rays_per_wave =
        requested_chunk_rays * static_cast<std::uint64_t>(threads);
    const std::uint64_t waves =
        std::max<std::uint64_t>(1U, (total_records + rays_per_wave - 1U) / rays_per_wave);
    const std::uint64_t item_count = std::min<std::uint64_t>(
        total_records, waves * static_cast<std::uint64_t>(threads));
    const std::uint64_t base_count = total_records / item_count;
    const std::uint64_t remainder = total_records % item_count;
    std::vector<WorkItem> result;
    result.reserve(static_cast<std::size_t>(item_count));
    std::uint64_t start = 0U;
    for (std::uint64_t index = 0U; index < item_count; ++index) {
        const std::uint64_t count = base_count + (index < remainder ? 1U : 0U);
        result.push_back({start, count});
        start += count;
    }
    return result;
}

void addFileRange(
    const InputFile& file, const std::uint64_t first_record, const std::uint64_t record_count,
    navvis_recon::MappedSpaceQualityGrid& grid) {
    std::ifstream input(file.path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + file.path.string());
    }
    input.seekg(static_cast<std::streamoff>(first_record * sizeof(DiskRecordV2)));
    if (!input) {
        throw std::runtime_error("cannot seek in " + file.path.string());
    }

    constexpr std::size_t kReadBlockRecords = 65'536U;
    std::vector<DiskRecordV2> buffer(kReadBlockRecords);
    std::uint64_t records_read = 0U;
    while (records_read < record_count) {
        const std::size_t block_records = static_cast<std::size_t>(
            std::min<std::uint64_t>(buffer.size(), record_count - records_read));
        const std::streamsize block_bytes =
            static_cast<std::streamsize>(block_records * sizeof(DiskRecordV2));
        input.read(reinterpret_cast<char*>(buffer.data()), block_bytes);
        if (input.gcount() != block_bytes) {
            throw std::runtime_error("read failure in " + file.path.string());
        }

        for (std::size_t index = 0U; index < block_records; ++index) {
            const DiskRecordV2& record = buffer[index];
            if (record.count != 1U) {
                throw std::runtime_error(
                    "mapped-space quality requires exact per-ray V2 shards; "
                    "clustered count != 1 in " + file.path.string());
            }

            navvis_recon::MappedSpaceQualityRay ray;
            ray.origin = Eigen::Vector3d(record.origin_x, record.origin_y, record.origin_z);
            ray.endpoint = Eigen::Vector3d(record.xyz_x, record.xyz_y, record.xyz_z);
            grid.addRayAtInputIndex(ray, static_cast<std::size_t>(
                file.global_start + first_record + records_read + index));
        }
        records_read += block_records;
    }
}

std::uint64_t addWorkItem(
    const std::vector<InputFile>& files, const WorkItem& item,
    navvis_recon::MappedSpaceQualityGrid& grid) {
    std::uint64_t current = item.global_start;
    const std::uint64_t end = item.global_start + item.records;
    for (const InputFile& file : files) {
        const std::uint64_t file_end = file.global_start + file.records;
        if (file_end <= current) {
            continue;
        }
        if (file.global_start >= end) {
            break;
        }
        const std::uint64_t first_record = current - file.global_start;
        const std::uint64_t count = std::min(end - current, file.records - first_record);
        addFileRange(file, first_record, count, grid);
        current += count;
        if (current == end) {
            break;
        }
    }
    if (current != end) {
        throw std::runtime_error("mapped-space work item exceeds available shard records");
    }
    return item.records;
}

int workerThreads(const Options& options, const std::size_t work_items) {
    int available = 1;
#ifdef _OPENMP
    available = std::max(1, omp_get_max_threads());
#endif
    const int requested = options.threads > 0
                              ? options.threads
                              : std::min(12, available);
    return std::max(1, std::min<int>(requested, static_cast<int>(work_items)));
}

std::uint64_t addShardsParallel(
    const std::vector<InputFile>& files, const Options& options,
    navvis_recon::MappedSpaceQualityGrid& grid) {
    const std::uint64_t total_records = totalRecords(files);
    const int threads = workerThreads(options, static_cast<std::size_t>(total_records));
    const std::vector<WorkItem> work = workItems(total_records, options.chunk_rays, threads);
    std::cerr << "Mapped-space aggregation: " << total_records << " rays in "
              << work.size() << " ordered chunks, " << threads << " OpenMP workers\n";

    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    std::uint64_t merged_records = 0U;
    for (std::size_t wave = 0U; wave < work.size(); wave += static_cast<std::size_t>(threads)) {
        const int wave_size = static_cast<int>(
            std::min<std::size_t>(static_cast<std::size_t>(threads), work.size() - wave));
        std::vector<std::unique_ptr<navvis_recon::MappedSpaceQualityGrid>> partials(
            static_cast<std::size_t>(wave_size));
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(wave_size));

#pragma omp parallel for schedule(static) num_threads(threads)
        for (int local = 0; local < wave_size; ++local) {
            try {
                auto partial =
                    std::make_unique<navvis_recon::MappedSpaceQualityGrid>(options.quality);
                addWorkItem(files, work[wave + static_cast<std::size_t>(local)], *partial);
                partials[static_cast<std::size_t>(local)] = std::move(partial);
            } catch (...) {
                errors[static_cast<std::size_t>(local)] = std::current_exception();
            }
        }

        for (int local = 0; local < wave_size; ++local) {
            if (errors[static_cast<std::size_t>(local)]) {
                std::rethrow_exception(errors[static_cast<std::size_t>(local)]);
            }
            merged_records += work[wave + static_cast<std::size_t>(local)].records;
            grid.mergeLaterChunk(std::move(*partials[static_cast<std::size_t>(local)]));
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        std::cerr << "Mapped-space chunks " << std::min(work.size(), wave + wave_size) << '/'
                  << work.size() << ", rays " << merged_records << '/' << total_records
                  << ", " << seconds << " s\r";
    }
    std::cerr << '\n';
    return total_records;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        navvis_recon::MappedSpaceQualityGrid grid(options.quality);
        const std::vector<InputFile> files = inputFiles(options.input_shards);
        const std::uint64_t records_read = addShardsParallel(files, options, grid);
        if (grid.inputRayCount() != records_read) {
            throw std::runtime_error("mapped-space parallel merge lost input rays");
        }
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
