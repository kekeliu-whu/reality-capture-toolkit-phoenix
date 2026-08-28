#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Record = std::array<std::uint32_t, 7>;

std::size_t propertySize(const std::string& type) {
    if (type == "float" || type == "uint" || type == "int") return 4;
    if (type == "double") return 8;
    if (type == "ushort" || type == "short") return 2;
    if (type == "uchar" || type == "char") return 1;
    throw std::runtime_error("unsupported PLY property type: " + type);
}

std::vector<Record> readRecords(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open: " + path);

    std::size_t count = 0;
    std::size_t stride = 0;
    std::unordered_map<std::string, std::size_t> offsets;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("element vertex ", 0) == 0) {
            count = std::stoull(line.substr(15));
        } else if (line.rfind("property ", 0) == 0) {
            const auto type_end = line.find(' ', 9);
            if (type_end == std::string::npos) {
                throw std::runtime_error("bad property line");
            }
            const std::string type = line.substr(9, type_end - 9);
            offsets.emplace(line.substr(type_end + 1), stride);
            stride += propertySize(type);
        } else if (line == "end_header") {
            break;
        }
    }

    const std::array<std::string, 7> fields = {
        "origin_x", "origin_y", "origin_z", "x", "y", "z", "intensity"};
    for (const auto& field : fields) {
        if (!offsets.count(field)) throw std::runtime_error("missing field: " + field);
    }

    std::vector<Record> records(count);
    std::vector<char> row(stride);
    for (auto& record : records) {
        stream.read(row.data(), static_cast<std::streamsize>(row.size()));
        if (!stream) throw std::runtime_error("short PLY payload: " + path);
        for (std::size_t index = 0; index < fields.size(); ++index) {
            std::memcpy(&record[index], row.data() + offsets.at(fields[index]),
                        sizeof(std::uint32_t));
        }
    }
    return records;
}

float asFloat(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void printRecord(const char* side, const Record& record) {
    std::cout << side;
    for (const auto bits : record) {
        std::cout << ' ' << std::setprecision(9) << asFloat(bits)
                  << "(0x" << std::hex << std::setw(8) << std::setfill('0') << bits
                  << std::dec << std::setfill(' ') << ')';
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::cerr << "usage: compare_ply_common_multiset reference.ply candidate.ply\n";
        return 2;
    }
    auto reference = readRecords(argv[1]);
    auto candidate = readRecords(argv[2]);
    std::sort(reference.begin(), reference.end());
    std::sort(candidate.begin(), candidate.end());

    std::size_t reference_index = 0;
    std::size_t candidate_index = 0;
    std::size_t common = 0;
    std::size_t reference_only = 0;
    std::size_t candidate_only = 0;
    constexpr std::size_t kPrintLimit = 20;
    while (reference_index < reference.size() &&
           candidate_index < candidate.size()) {
        if (reference[reference_index] == candidate[candidate_index]) {
            ++common;
            ++reference_index;
            ++candidate_index;
        } else if (reference[reference_index] < candidate[candidate_index]) {
            if (reference_only < kPrintLimit) {
                printRecord("reference_only", reference[reference_index]);
            }
            ++reference_only;
            ++reference_index;
        } else {
            if (candidate_only < kPrintLimit) {
                printRecord("candidate_only", candidate[candidate_index]);
            }
            ++candidate_only;
            ++candidate_index;
        }
    }
    while (reference_index < reference.size()) {
        if (reference_only < kPrintLimit) {
            printRecord("reference_only", reference[reference_index]);
        }
        ++reference_only;
        ++reference_index;
    }
    while (candidate_index < candidate.size()) {
        if (candidate_only < kPrintLimit) {
            printRecord("candidate_only", candidate[candidate_index]);
        }
        ++candidate_only;
        ++candidate_index;
    }

    std::cout << "reference_records " << reference.size() << '\n'
              << "candidate_records " << candidate.size() << '\n'
              << "common_records " << common << '\n'
              << "reference_only " << reference_only << '\n'
              << "candidate_only " << candidate_only << '\n'
              << "common_field_multiset_exact "
              << (reference_only == 0 && candidate_only == 0) << '\n';
    return reference_only == 0 && candidate_only == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
