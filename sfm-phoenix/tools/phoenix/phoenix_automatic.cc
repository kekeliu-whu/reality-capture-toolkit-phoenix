#include "phoenix_tool.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace phoenix_tool
{

  bool HasMapperOption(const std::vector<std::string> &mapper_args,
                       const std::string &option)
  {
    for (const auto &arg : mapper_args)
    {
      if (arg == option)
      {
        return true;
      }
    }
    return false;
  }

  AutomaticOptions ParseAutomaticOptions(int argc, char **argv)
  {
    AutomaticOptions options;

    const std::unordered_set<std::string> handled_flags = {
        "--database_path",
        "--image_path",
        "--output_path",
        "--camera_mode",
        "--ImageReader.camera_model",
        "--Phoenix.max_edge",
        "--Phoenix.top_k",
        "--Phoenix.max_matches",
        "--Phoenix.retrieval_num",
        "--Phoenix.retrieval_batch_size",
        "--Phoenix.retrieval_similarity_threshold",
        "--Phoenix.retrieval_relative_threshold",
        "--Phoenix.filter_static_frames",
        "--Phoenix.single_camera_per_folder",
        "--Phoenix.static_frame_diff_threshold",
        "--Phoenix.linear_overlap_num",
        "--Phoenix.quadratic_overlap_num",
        "--Phoenix.skip_existing",
        "--Phoenix.overwrite_existing",
    };

    // Pre-expand "--key=value" into separate "--key" / "value" tokens so that
    // both "  --key value" and "--key=value" forms are handled uniformly.
    std::vector<std::string> expanded_args;
    for (int i = 2; i < argc; ++i)
    {
      const std::string tok = argv[i];
      const auto eq = tok.find('=');
      if (eq != std::string::npos && tok.substr(0, 2) == "--")
      {
        expanded_args.push_back(tok.substr(0, eq));
        expanded_args.push_back(tok.substr(eq + 1));
      }
      else
      {
        expanded_args.push_back(tok);
      }
    }

    for (int index = 0; index < static_cast<int>(expanded_args.size());
         ++index)
    {
      const std::string &arg = expanded_args[index];
      const bool has_value = index + 1 < static_cast<int>(expanded_args.size());
      if (arg == "--database_path" && has_value)
      {
        options.database_path = expanded_args[++index];
      }
      else if (arg == "--image_path" && has_value)
      {
        options.image_path = expanded_args[++index];
      }
      else if (arg == "--output_path" && has_value)
      {
        options.output_path = expanded_args[++index];
      }
      else if (arg == "--camera_mode" && has_value)
      {
        options.extraction.camera_mode = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--ImageReader.camera_model" && has_value)
      {
        options.extraction.camera_model = expanded_args[++index];
      }
      else if (arg == "--Phoenix.max_edge" && has_value)
      {
        options.extraction.max_edge = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.top_k" && has_value)
      {
        options.extraction.top_k = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.max_matches" && has_value)
      {
        options.matching.max_matches = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.retrieval_num" && has_value)
      {
        options.matching.retrieval.num = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.retrieval_batch_size" && has_value)
      {
        options.matching.retrieval.batch_size = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.retrieval_similarity_threshold" &&
               has_value)
      {
        options.matching.retrieval.similarity_threshold = std::stod(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.retrieval_relative_threshold" && has_value)
      {
        options.matching.retrieval.relative_threshold =
            std::stod(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.filter_static_frames" && has_value)
      {
        options.extraction.filter_static_frames = ParseBool(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.single_camera_per_folder" && has_value)
      {
        options.extraction.single_camera_per_folder = ParseBool(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.static_frame_diff_threshold" && has_value)
      {
        options.extraction.static_frame_diff_threshold = std::stod(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.linear_overlap_num" && has_value)
      {
        options.matching.linear_overlap_num = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.quadratic_overlap_num" && has_value)
      {
        options.matching.quadratic_overlap_num = std::stoi(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.skip_existing" && has_value)
      {
        options.extraction.skip_existing = options.matching.skip_existing = ParseBool(expanded_args[++index]);
      }
      else if (arg == "--Phoenix.overwrite_existing" && has_value)
      {
        options.matching.overwrite_existing = ParseBool(expanded_args[++index]);
      }
      else
      {
        options.mapper_args.push_back(arg);
        if (arg.rfind("--", 0) == 0 && has_value &&
            handled_flags.find(arg) == handled_flags.end() &&
            expanded_args[index + 1].rfind("--", 0) != 0)
        {
          options.mapper_args.push_back(expanded_args[++index]);
        }
      }
    }

    if (options.database_path.empty() || options.image_path.empty() ||
        options.output_path.empty())
    {
      throw std::runtime_error(
          "automatic_reconstructor requires --database_path, --image_path, "
          "and --output_path");
    }
    if (options.matching.linear_overlap_num < 0 ||
        options.matching.quadratic_overlap_num < 0)
    {
      throw std::runtime_error("Phoenix overlap params must be non-negative");
    }
    if (options.extraction.static_frame_diff_threshold < 0.0)
    {
      throw std::runtime_error(
          "Phoenix static_frame_diff_threshold must be non-negative");
    }
    if (options.matching.retrieval.similarity_threshold < 0.0 ||
        options.matching.retrieval.similarity_threshold > 1.0)
    {
      throw std::runtime_error(
          "Phoenix retrieval_similarity_threshold must be in [0, 1]");
    }
    if (options.matching.retrieval.relative_threshold < 0.0 ||
        options.matching.retrieval.relative_threshold > 1.0)
    {
      throw std::runtime_error(
          "Phoenix retrieval_relative_threshold must be in [0, 1]");
    }
    return options;
  }

  int RunAutomaticReconstructor(const AutomaticOptions &options)
  {
    const auto& ext = options.extraction;
    const auto& mat = options.matching;
    const auto& ret = mat.retrieval;

    spdlog::info("[automatic_reconstructor] options:");
    spdlog::info("  database_path              = {}", options.database_path);
    spdlog::info("  image_path                 = {}", options.image_path);
    spdlog::info("  output_path                = {}", options.output_path);
    spdlog::info("  camera_mode                = {}", ext.camera_mode);
    spdlog::info("  camera_model               = {}",
                 ext.camera_model.empty() ? "(default)" : ext.camera_model);
    spdlog::info("  max_edge                   = {}", ext.max_edge);
    spdlog::info("  top_k                      = {}", ext.top_k);
    spdlog::info("  max_matches                = {}", mat.max_matches);
    spdlog::info("  retrieval_num              = {}", ret.num);
    spdlog::info("  retrieval_batch_size       = {}", ret.batch_size);
    spdlog::info("  retrieval_similarity_threshold = {:.4f}",
                 ret.similarity_threshold);
    spdlog::info("  retrieval_relative_threshold   = {:.4f}",
                 ret.relative_threshold);
    spdlog::info("  linear_overlap_num         = {}", mat.linear_overlap_num);
    spdlog::info("  quadratic_overlap_num      = {}",
                 mat.quadratic_overlap_num);
    spdlog::info("  filter_static_frames       = {}", ext.filter_static_frames);
    spdlog::info("  single_camera_per_folder   = {}",
                 ext.single_camera_per_folder);
    spdlog::info("  static_frame_diff_threshold= {:.4f}",
                 ext.static_frame_diff_threshold);
    spdlog::info("  skip_existing              = {} / {}",
                 ext.skip_existing, mat.skip_existing);
    spdlog::info("  overwrite_existing         = {}", mat.overwrite_existing);
    if (!options.mapper_args.empty())
    {
      std::string extra;
      for (const auto &a : options.mapper_args)
      {
        if (!extra.empty())
          extra += ' ';
        extra += a;
      }
      spdlog::info("  mapper_args                = {}", extra);
    }

    // Feature extraction — pass options.extraction directly.
    int exit_code = ExecFeatureExtractor(
        options.database_path, options.image_path, options.extraction);
    if (exit_code != 0)
    {
      return exit_code;
    }

    // Feature matching — set image_path for retrieval if not overridden.
    FeatureMatchingOptions matching_opts = options.matching;
    if (matching_opts.image_path.empty())
    {
      matching_opts.image_path = options.image_path;
    }
    exit_code = ExecFeatureMatcher(options.database_path, matching_opts);
    if (exit_code != 0)
    {
      return exit_code;
    }

    std::vector<std::string> mapper_args = {
        "--database_path",
        options.database_path,
        "--image_path",
        options.image_path,
        "--output_path",
        options.output_path,
    };
    mapper_args.insert(mapper_args.end(),
                       options.mapper_args.begin(),
                       options.mapper_args.end());

    const bool has_multiple_models =
      HasMapperOption(mapper_args, "--Mapper.multiple_models");
    const bool has_max_num_models =
      HasMapperOption(mapper_args, "--Mapper.max_num_models");
    if (!has_multiple_models && !has_max_num_models)
    {
      mapper_args.push_back("--Mapper.multiple_models");
      mapper_args.push_back("0");
      mapper_args.push_back("--Mapper.max_num_models");
      mapper_args.push_back("1");
      spdlog::info(
        "  mapper default override      = --Mapper.multiple_models 0 "
        "--Mapper.max_num_models 1");
    }

    std::filesystem::create_directories(options.output_path);
    return RunCommand("colmap", mapper_args, "mapper");
  }

} // namespace phoenix_tool