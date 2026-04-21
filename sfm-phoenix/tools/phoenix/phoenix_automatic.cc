#include "phoenix_tool.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace phoenix_tool {

AutomaticOptions ParseAutomaticOptions(int argc, char** argv) {
  AutomaticOptions options;
  const std::unordered_set<std::string> handled_flags = {
      "--database_path",
      "--image_path",
      "--output_path",
      "--image_list_path",
      "--pair_list_path",
      "--colmap_path",
      "--Phoenix.retrieval_model_path",
      "--camera_mode",
      "--Phoenix.max_edge",
      "--Phoenix.top_k",
      "--Phoenix.max_matches",
      "--Phoenix.retrieval_num",
      "--Phoenix.retrieval_batch_size",
      "--Phoenix.filter_static_frames",
      "--Phoenix.static_frame_diff_threshold",
      "--Phoenix.linear_overlap_num",
      "--Phoenix.quadratic_overlap_num",
      "--Phoenix.skip_existing",
      "--Phoenix.overwrite_existing",
  };

  // Pre-expand "--key=value" into separate "--key" / "value" tokens so that
  // both "  --key value" and "--key=value" forms are handled uniformly.
  std::vector<std::string> expanded_args;
  for (int i = 2; i < argc; ++i) {
    const std::string tok = argv[i];
    const auto eq = tok.find('=');
    if (eq != std::string::npos && tok.substr(0, 2) == "--") {
      expanded_args.push_back(tok.substr(0, eq));
      expanded_args.push_back(tok.substr(eq + 1));
    } else {
      expanded_args.push_back(tok);
    }
  }

  for (int index = 0; index < static_cast<int>(expanded_args.size());
       ++index) {
    const std::string& arg = expanded_args[index];
    const bool has_value = index + 1 < static_cast<int>(expanded_args.size());
    if (arg == "--database_path" && has_value) {
      options.database_path = expanded_args[++index];
    } else if (arg == "--image_path" && has_value) {
      options.image_path = expanded_args[++index];
    } else if (arg == "--output_path" && has_value) {
      options.output_path = expanded_args[++index];
    } else if (arg == "--image_list_path" && has_value) {
      options.image_list_path = expanded_args[++index];
    } else if (arg == "--pair_list_path" && has_value) {
      options.pair_list_path = expanded_args[++index];
    } else if (arg == "--colmap_path" && has_value) {
      options.colmap_path = expanded_args[++index];
    } else if (arg == "--Phoenix.retrieval_model_path" && has_value) {
      options.retrieval_model_path = expanded_args[++index];
    } else if (arg == "--camera_mode" && has_value) {
      options.camera_mode = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.max_edge" && has_value) {
      options.max_edge = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.top_k" && has_value) {
      options.top_k = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.max_matches" && has_value) {
      options.max_matches = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.retrieval_num" && has_value) {
      options.retrieval_num = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.retrieval_batch_size" && has_value) {
      options.retrieval_batch_size = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.filter_static_frames" && has_value) {
      options.filter_static_frames = ParseBool(expanded_args[++index]);
    } else if (arg == "--Phoenix.static_frame_diff_threshold" && has_value) {
      options.static_frame_diff_threshold = std::stod(expanded_args[++index]);
    } else if (arg == "--Phoenix.linear_overlap_num" && has_value) {
      options.linear_overlap_num = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.quadratic_overlap_num" && has_value) {
      options.quadratic_overlap_num = std::stoi(expanded_args[++index]);
    } else if (arg == "--Phoenix.skip_existing" && has_value) {
      options.skip_existing = ParseBool(expanded_args[++index]);
    } else if (arg == "--Phoenix.overwrite_existing" && has_value) {
      options.overwrite_existing = ParseBool(expanded_args[++index]);
    } else {
      options.mapper_args.push_back(arg);
      if (arg.rfind("--", 0) == 0 && has_value &&
          handled_flags.find(arg) == handled_flags.end() &&
          expanded_args[index + 1].rfind("--", 0) != 0) {
        options.mapper_args.push_back(expanded_args[++index]);
      }
    }
  }

  if (options.database_path.empty() || options.image_path.empty() ||
      options.output_path.empty()) {
    throw std::runtime_error(
        "automatic_reconstructor requires --database_path, --image_path, "
        "and --output_path");
  }
  if (options.linear_overlap_num < 0 || options.quadratic_overlap_num < 0) {
    throw std::runtime_error("Phoenix overlap params must be non-negative");
  }
  if (options.static_frame_diff_threshold < 0.0) {
    throw std::runtime_error(
        "Phoenix static_frame_diff_threshold must be non-negative");
  }
  return options;
}

int RunAutomaticReconstructor(const AutomaticOptions& options) {
  std::vector<std::string> extractor_args = {
      "--database_path", options.database_path,
      "--image_path", options.image_path,
      "--Phoenix.max_edge", std::to_string(options.max_edge),
      "--Phoenix.top_k", std::to_string(options.top_k),
      "--Phoenix.filter_static_frames",
      options.filter_static_frames ? "true" : "false",
      "--Phoenix.static_frame_diff_threshold",
      std::to_string(options.static_frame_diff_threshold),
      "--Phoenix.skip_existing",
      options.skip_existing ? "true" : "false",
  };
  if (!options.image_list_path.empty()) {
    extractor_args.insert(extractor_args.end(),
                          {"--image_list_path", options.image_list_path});
  }
  if (options.camera_mode >= 0) {
    extractor_args.insert(extractor_args.end(),
                          {"--camera_mode",
                           std::to_string(options.camera_mode)});
  }

  std::vector<std::string> extractor_argv = {"feature_extractor"};
  extractor_argv.insert(extractor_argv.end(),
                        extractor_args.begin(),
                        extractor_args.end());
  std::vector<char*> extractor_argp;
  extractor_argp.reserve(extractor_argv.size());
  for (auto& arg : extractor_argv) {
    extractor_argp.push_back(arg.data());
  }

  int exit_code =
      RunFeatureExtractor(static_cast<int>(extractor_argp.size()),
                          extractor_argp.data());
  if (exit_code != 0) {
    return exit_code;
  }

  std::vector<std::string> matcher_args = {
      "--database_path", options.database_path,
      "--image_path", options.image_path,
      "--Phoenix.max_edge", std::to_string(options.max_edge),
      "--Phoenix.max_matches", std::to_string(options.max_matches),
      "--Phoenix.retrieval_num", std::to_string(options.retrieval_num),
      "--Phoenix.retrieval_batch_size",
      std::to_string(options.retrieval_batch_size),
      "--Phoenix.linear_overlap_num",
      std::to_string(options.linear_overlap_num),
      "--Phoenix.quadratic_overlap_num",
      std::to_string(options.quadratic_overlap_num),
      "--Phoenix.skip_existing",
      options.skip_existing ? "true" : "false",
      "--Phoenix.overwrite_existing",
      options.overwrite_existing ? "true" : "false",
  };
  if (!options.retrieval_model_path.empty()) {
    matcher_args.insert(matcher_args.end(),
                        {"--Phoenix.retrieval_model_path",
                         options.retrieval_model_path});
  }
  if (!options.image_list_path.empty()) {
    matcher_args.insert(matcher_args.end(),
                        {"--image_list_path", options.image_list_path});
  }
  if (!options.pair_list_path.empty()) {
    matcher_args.insert(matcher_args.end(),
                        {"--pair_list_path", options.pair_list_path});
  }

  std::vector<std::string> matcher_argv = {"feature_matcher"};
  matcher_argv.insert(matcher_argv.end(),
                      matcher_args.begin(),
                      matcher_args.end());
  std::vector<char*> matcher_argp;
  matcher_argp.reserve(matcher_argv.size());
  for (auto& arg : matcher_argv) {
    matcher_argp.push_back(arg.data());
  }

  exit_code =
      RunFeatureMatcher(static_cast<int>(matcher_argp.size()),
                        matcher_argp.data());
  if (exit_code != 0) {
    return exit_code;
  }

  std::vector<std::string> mapper_args = {
      "--database_path", options.database_path,
      "--image_path", options.image_path,
      "--output_path", options.output_path,
  };
  mapper_args.insert(mapper_args.end(),
                     options.mapper_args.begin(),
                     options.mapper_args.end());
  std::filesystem::create_directories(options.output_path);
  return RunCommand(options.colmap_path, mapper_args, "mapper");
}

}  // namespace phoenix_tool