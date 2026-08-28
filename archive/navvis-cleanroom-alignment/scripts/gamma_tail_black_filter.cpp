#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSlots = 5;

#pragma pack(push, 1)
struct ColoredPoint {
    float x;
    float y;
    float z;
    std::uint8_t red;
    std::uint8_t green;
    std::uint8_t blue;
    std::uint8_t alpha;
    float intensity;
    float nx;
    float ny;
    float nz;
    float curvature;
};
#pragma pack(pop)
static_assert(sizeof(ColoredPoint) == 36);

struct GammaModel {
    double gain = 1.0;
    double exponent = 1.0;
};

struct Observation {
    std::array<std::uint8_t, 3> rgb{};
    std::uint16_t quality = 0;
    int view = -1;
};

struct Candidate {
    bool valid = false;
    std::array<std::uint8_t, 3> rgb{};
};

struct Metrics {
    std::uint64_t points = 0;
    std::uint64_t exact_points = 0;
    std::uint64_t within_one_points = 0;
    std::uint64_t absolute_sum = 0;
    std::uint64_t changed_channels = 0;
    int maximum_delta = 0;

    void add(const Candidate& candidate, const ColoredPoint& reference) {
        if (!candidate.valid) {
            return;
        }
        const std::array<std::uint8_t, 3> expected{reference.red, reference.green, reference.blue};
        bool exact = true;
        bool within_one = true;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const int delta = std::abs(static_cast<int>(candidate.rgb[channel]) -
                                       static_cast<int>(expected[channel]));
            absolute_sum += static_cast<std::uint64_t>(delta);
            changed_channels += delta != 0;
            maximum_delta = std::max(maximum_delta, delta);
            exact &= delta == 0;
            within_one &= delta <= 1;
        }
        ++points;
        exact_points += exact;
        within_one_points += within_one;
    }
};

struct VariantMetrics {
    Metrics all;
    Metrics any_black;
    Metrics no_black;
    std::uint64_t invalid_after_filter = 0;
};

std::uint64_t parsePlyOffset(std::ifstream& stream, const std::string& path) {
    std::string line;
    std::uint64_t count = 0;
    bool ended = false;
    while (std::getline(stream, line)) {
        if (line.rfind("element vertex ", 0) == 0) {
            count = std::stoull(line.substr(std::strlen("element vertex ")));
        }
        if (line == "end_header") {
            ended = true;
            break;
        }
    }
    if (!ended || count == 0) {
        throw std::runtime_error("invalid PLY header: " + path);
    }
    return count;
}

std::vector<GammaModel> loadModels(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open GammaModels: " + path);
    }
    std::vector<GammaModel> models(136);
    std::vector<bool> seen(models.size(), false);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        std::size_t view = 0;
        GammaModel model;
        if (!(fields >> view >> model.gain >> model.exponent) || view >= models.size()) {
            throw std::runtime_error("invalid GammaModel line: " + line);
        }
        models[view] = model;
        seen[view] = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::runtime_error("GammaModel coverage is incomplete");
    }
    return models;
}

std::array<float, 3> applyGamma(const Observation& observation, const GammaModel& model) {
    // Reproduce the two wrapper/core pairs at 0x23c450/0x256140 and
    // 0x23a9c0/0x256780. The apparently redundant /255 and *255 operations
    // are observable at final byte boundaries and therefore stay explicit.
    constexpr float scale255 = 255.0F;
    constexpr float degrees_per_turn = 360.0F;
    constexpr float degrees_per_sector = 60.0F;
    const float red = static_cast<float>(observation.rgb[0]) / scale255;
    const float green = static_cast<float>(observation.rgb[1]) / scale255;
    const float blue = static_cast<float>(observation.rgb[2]) / scale255;
    const float maximum = std::max({red, green, blue});
    const float minimum = std::min({red, green, blue});
    const float delta = maximum - minimum;

    float hue = 0.0F;
    float saturation = 0.0F;
    if (maximum > 0.0F) {
        saturation = delta / maximum;
    }
    if (delta > 0.0F) {
        float degrees = 0.0F;
        if (maximum == red) {
            degrees = (green - blue) / delta;
            if (blue > green) {
                degrees += 6.0F;
            }
        } else if (maximum == green) {
            degrees = (blue - red) / delta + 2.0F;
        } else {
            degrees = (red - green) / delta + 4.0F;
        }
        degrees *= degrees_per_sector;
        hue = degrees * (scale255 / degrees_per_turn);
    }
    saturation *= scale255;
    const float original_value = maximum * scale255;

    const float normalized = original_value / scale255;
    const float corrected = static_cast<float>(
        model.gain * std::pow(static_cast<double>(normalized), model.exponent));
    const float value255 = corrected * scale255;
    if (saturation <= 0.0F) {
        return {value255, value255, value255};
    }

    float degrees = hue * degrees_per_turn;
    degrees /= scale255;
    const float normalized_saturation = saturation / scale255;
    const float normalized_value = value255 / scale255;
    const float chroma = normalized_saturation * normalized_value;
    const float minimum_value = normalized_value - chroma;
    const float turns = degrees / degrees_per_turn;
    const float wrapped_degrees = degrees - std::floor(turns) * degrees_per_turn;
    const float sector_value = wrapped_degrees / degrees_per_sector;
    const float twice_floor_half = std::floor(0.5F * sector_value) * 2.0F;
    const float triangle =
        1.0F - std::fabs(sector_value - twice_floor_half - 1.0F);
    const float x = triangle * chroma;
    const int sector = static_cast<int>(sector_value);
    std::array<float, 3> normalized_rgb{};
    switch (sector % 6) {
        case 0: normalized_rgb = {chroma + minimum_value, x + minimum_value, minimum_value}; break;
        case 1: normalized_rgb = {x + minimum_value, chroma + minimum_value, minimum_value}; break;
        case 2: normalized_rgb = {minimum_value, chroma + minimum_value, x + minimum_value}; break;
        case 3: normalized_rgb = {minimum_value, x + minimum_value, chroma + minimum_value}; break;
        case 4: normalized_rgb = {x + minimum_value, minimum_value, chroma + minimum_value}; break;
        default: normalized_rgb = {chroma + minimum_value, minimum_value, x + minimum_value}; break;
    }
    for (float& channel : normalized_rgb) {
        channel *= scale255;
    }
    return normalized_rgb;
}

std::uint8_t roundByte(float value) {
    const long rounded = std::lrintf(value);
    return static_cast<std::uint8_t>(std::clamp<long>(rounded, 0, 255));
}

Candidate blend(const std::array<Observation, kSlots>& input, std::size_t count,
                const std::vector<GammaModel>& models, bool filter_black,
                bool binary_double_normalization, bool clamp_success_min_one = false) {
    std::array<Observation, kSlots> observations{};
    std::size_t retained = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const bool black = input[index].rgb[0] == 0 && input[index].rgb[1] == 0 &&
                           input[index].rgb[2] == 0;
        if (filter_black && black) {
            continue;
        }
        observations[retained++] = input[index];
    }
    if (retained == 0) {
        return {};
    }

    std::array<std::array<float, 3>, kSlots> colors{};
    std::array<float, kSlots> weights{};
    float minimum_cost = 1.0F;
    for (std::size_t index = 0; index < retained; ++index) {
        colors[index] = applyGamma(observations[index], models.at(observations[index].view));
        const float score = static_cast<float>(observations[index].quality) / 65535.0F;
        weights[index] = 1.0F - score;
        minimum_cost = std::min(minimum_cost, weights[index]);
    }
    const float bandwidth = std::max(1.0e-6F, 0.1F * minimum_cost);
    for (std::size_t index = 0; index < retained; ++index) {
        weights[index] = std::exp(-weights[index] / bandwidth);
    }

    if (binary_double_normalization) {
        double normalizer = 0.0;
        for (std::size_t index = 0; index < retained; ++index) {
            normalizer += std::fabs(static_cast<double>(weights[index]));
        }
        for (std::size_t index = 0; index < retained; ++index) {
            weights[index] = static_cast<float>(static_cast<double>(weights[index]) / normalizer);
        }
        double caller_normalizer = 0.0;
        for (std::size_t index = 0; index < retained; ++index) {
            caller_normalizer += std::fabs(static_cast<double>(weights[index]));
        }
        for (std::size_t index = 0; index < retained; ++index) {
            weights[index] =
                static_cast<float>(static_cast<double>(weights[index]) / caller_normalizer);
        }
    } else {
        float normalizer = 0.0F;
        for (std::size_t index = 0; index < retained; ++index) {
            normalizer += weights[index];
        }
        for (std::size_t index = 0; index < retained; ++index) {
            weights[index] /= normalizer;
        }
    }

    std::array<float, 3> accumulated{};
    float total_weight = 0.0F;
    for (std::size_t index = 0; index < retained; ++index) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            accumulated[channel] += weights[index] * colors[index][channel];
        }
        total_weight += weights[index];
    }
    Candidate candidate;
    candidate.valid = true;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        candidate.rgb[channel] = roundByte(accumulated[channel] / total_weight);
        if (clamp_success_min_one) {
            candidate.rgb[channel] = std::max<std::uint8_t>(1, candidate.rgb[channel]);
        }
    }
    return candidate;
}

void emitMetrics(std::ostream& output, const Metrics& metrics) {
    output << "{\"points\":" << metrics.points
           << ",\"rgb_mae\":"
           << (metrics.points == 0 ? 0.0
                                   : static_cast<double>(metrics.absolute_sum) /
                                         static_cast<double>(metrics.points * 3))
           << ",\"exact_point_fraction\":"
           << (metrics.points == 0 ? 0.0
                                   : static_cast<double>(metrics.exact_points) / metrics.points)
           << ",\"max_channel_le_1_fraction\":"
           << (metrics.points == 0
                   ? 0.0
                   : static_cast<double>(metrics.within_one_points) / metrics.points)
           << ",\"changed_channels\":" << metrics.changed_channels
           << ",\"maximum_delta\":" << metrics.maximum_delta << '}';
}

void emitVariant(std::ostream& output, const VariantMetrics& variant) {
    output << "{\"all\":";
    emitMetrics(output, variant.all);
    output << ",\"any_black_observation\":";
    emitMetrics(output, variant.any_black);
    output << ",\"no_black_observation\":";
    emitMetrics(output, variant.no_black);
    output << ",\"invalid_after_filter\":" << variant.invalid_after_filter << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "usage: gamma_tail_black_filter GAMMA_MODELS FINAL_OVS VENDOR_PLY "
                         "[MISMATCH_CSV]\n";
            return 2;
        }
        std::fesetround(FE_TONEAREST);
        const std::vector<GammaModel> models = loadModels(argv[1]);
        std::ifstream ovs(argv[2], std::ios::binary);
        std::ifstream ply(argv[3], std::ios::binary);
        if (!ovs || !ply) {
            throw std::runtime_error("cannot open OVS or PLY input");
        }
        const std::uint64_t points = parsePlyOffset(ply, argv[3]);
        ovs.seekg(0, std::ios::end);
        const auto ovs_size = ovs.tellg();
        ovs.seekg(0);
        if (ovs_size != static_cast<std::streamoff>(points * kSlots * 8)) {
            throw std::runtime_error("OVS size does not match PLY point count");
        }

        VariantMetrics keep_float;
        VariantMetrics keep_double;
        VariantMetrics filter_float;
        VariantMetrics filter_double;
        VariantMetrics official_direct;
        std::uint64_t official_direct_points = 0;
        std::uint64_t official_fallback_points = 0;
        std::uint64_t valid_observations = 0;
        std::uint64_t black_observations = 0;
        std::uint64_t points_with_black = 0;
        std::uint64_t direct_points = 0;
        std::array<Candidate, 4> probe_40595{};
        std::ofstream mismatch_csv;
        if (argc == 5) {
            mismatch_csv.open(argv[4]);
            if (!mismatch_csv) {
                throw std::runtime_error("cannot create mismatch CSV");
            }
            mismatch_csv << "point,any_black,observation_count,reference_rgb,"
                            "keep_float_rgb,keep_binary_rgb,filter_float_rgb,filter_binary_rgb,"
                            "filter_binary_max_delta,observations\n";
        }

        for (std::uint64_t point_index = 0; point_index < points; ++point_index) {
            std::array<std::uint8_t, kSlots * 8> packed{};
            ColoredPoint reference{};
            ovs.read(reinterpret_cast<char*>(packed.data()), packed.size());
            ply.read(reinterpret_cast<char*>(&reference), sizeof(reference));
            if (!ovs || !ply) {
                throw std::runtime_error("short read in OVS or PLY body");
            }
            std::array<Observation, kSlots> observations{};
            std::size_t count = 0;
            bool any_black = false;
            std::size_t non_black_count = 0;
            for (std::size_t slot = 0; slot < kSlots; ++slot) {
                const std::size_t offset = slot * 8;
                const std::uint16_t quality = static_cast<std::uint16_t>(packed[offset + 6]) |
                                              (static_cast<std::uint16_t>(packed[offset + 7]) << 8);
                if (quality == 0) {
                    continue;
                }
                const unsigned capture = (static_cast<unsigned>(packed[offset]) << 8) |
                                         static_cast<unsigned>(packed[offset + 1]);
                const int view = static_cast<int>(capture * 4 + packed[offset + 2]);
                if (view < 0 || view >= static_cast<int>(models.size())) {
                    continue;
                }
                Observation observation;
                observation.rgb = {packed[offset + 3], packed[offset + 4], packed[offset + 5]};
                observation.quality = quality;
                observation.view = view;
                const bool black = observation.rgb[0] == 0 && observation.rgb[1] == 0 &&
                                   observation.rgb[2] == 0;
                any_black |= black;
                non_black_count += !black;
                black_observations += black;
                ++valid_observations;
                observations[count++] = observation;
            }
            if (count == 0) {
                continue;
            }
            ++direct_points;
            points_with_black += any_black;
            const std::array<Candidate, 4> candidates{
                blend(observations, count, models, false, false),
                blend(observations, count, models, false, true),
                blend(observations, count, models, true, false),
                blend(observations, count, models, true, true),
            };
            const bool official_is_direct = !(any_black && non_black_count <= 1);
            const Candidate official_candidate =
                blend(observations, count, models, true, true, true);
            if (official_is_direct) {
                ++official_direct_points;
                official_direct.all.add(official_candidate, reference);
                (any_black ? official_direct.any_black : official_direct.no_black)
                    .add(official_candidate, reference);
            } else {
                ++official_fallback_points;
            }
            std::array<VariantMetrics*, 4> variants{
                &keep_float, &keep_double, &filter_float, &filter_double};
            for (std::size_t variant = 0; variant < variants.size(); ++variant) {
                if (!candidates[variant].valid) {
                    ++variants[variant]->invalid_after_filter;
                    continue;
                }
                variants[variant]->all.add(candidates[variant], reference);
                (any_black ? variants[variant]->any_black : variants[variant]->no_black)
                    .add(candidates[variant], reference);
            }
            if (point_index == 40595) {
                probe_40595 = candidates;
            }
            if (mismatch_csv && candidates[3].valid) {
                const std::array<std::uint8_t, 3> expected{
                    reference.red, reference.green, reference.blue};
                int maximum_delta = 0;
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    maximum_delta = std::max(
                        maximum_delta,
                        std::abs(static_cast<int>(candidates[3].rgb[channel]) -
                                 static_cast<int>(expected[channel])));
                }
                if (maximum_delta != 0) {
                    const auto emit_rgb = [&](const Candidate& candidate) {
                        if (!candidate.valid) {
                            mismatch_csv << "NA";
                            return;
                        }
                        mismatch_csv << static_cast<int>(candidate.rgb[0]) << ':'
                                     << static_cast<int>(candidate.rgb[1]) << ':'
                                     << static_cast<int>(candidate.rgb[2]);
                    };
                    mismatch_csv << point_index << ',' << static_cast<int>(any_black) << ','
                                 << count << ',' << static_cast<int>(expected[0]) << ':'
                                 << static_cast<int>(expected[1]) << ':'
                                 << static_cast<int>(expected[2]) << ',';
                    for (const Candidate& candidate : candidates) {
                        emit_rgb(candidate);
                        mismatch_csv << ',';
                    }
                    mismatch_csv << maximum_delta << ',';
                    for (std::size_t index = 0; index < count; ++index) {
                        if (index != 0) {
                            mismatch_csv << ';';
                        }
                        mismatch_csv << observations[index].view << ':'
                                     << static_cast<int>(observations[index].rgb[0]) << ':'
                                     << static_cast<int>(observations[index].rgb[1]) << ':'
                                     << static_cast<int>(observations[index].rgb[2]) << ':'
                                     << observations[index].quality;
                    }
                    mismatch_csv << '\n';
                }
            }
        }

        std::cout << std::setprecision(17)
                  << "{\n  \"points\":" << points
                  << ",\n  \"direct_points\":" << direct_points
                  << ",\n  \"valid_observations\":" << valid_observations
                  << ",\n  \"black_observations\":" << black_observations
                  << ",\n  \"points_with_black_observation\":" << points_with_black
                  << ",\n  \"variants\":{\n    \"keep_black_float_abs\":";
        emitVariant(std::cout, keep_float);
        std::cout << ",\n    \"keep_black_binary_abs\":";
        emitVariant(std::cout, keep_double);
        std::cout << ",\n    \"filter_black_float_abs\":";
        emitVariant(std::cout, filter_float);
        std::cout << ",\n    \"filter_black_binary_abs\":";
        emitVariant(std::cout, filter_double);
        std::cout << ",\n    \"official_direct_chain\":";
        emitVariant(std::cout, official_direct);
        std::cout << "\n  },\n  \"official_direct_points\":" << official_direct_points
                  << ",\n  \"official_fallback_from_black_min2\":" << official_fallback_points
                  << ",\n  \"point_40595_candidates\":[";
        for (std::size_t variant = 0; variant < probe_40595.size(); ++variant) {
            if (variant != 0) {
                std::cout << ',';
            }
            const Candidate& candidate = probe_40595[variant];
            if (!candidate.valid) {
                std::cout << "null";
            } else {
                std::cout << '[' << static_cast<int>(candidate.rgb[0]) << ','
                          << static_cast<int>(candidate.rgb[1]) << ','
                          << static_cast<int>(candidate.rgb[2]) << ']';
            }
        }
        std::cout << "]\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gamma_tail_black_filter: " << error.what() << '\n';
        return 1;
    }
}
