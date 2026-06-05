#include "phoenix/phoenix_tool.h"

#include <migration/logging.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  try {
    InitSpdLog();
    phoenix_tool::InstallInterruptHandler();

    if (argc < 2) {
      phoenix_tool::PrintHelp();
      return EXIT_FAILURE;
    }

    const std::string command = argv[1];
    if (command == "help" || command == "-h" || command == "--help") {
      phoenix_tool::PrintHelp();
      return EXIT_SUCCESS;
    }

    if (command == "feature_extractor") {
      return phoenix_tool::RunFeatureExtractor(argc - 1, argv + 1);
    }

    if (command == "feature_matcher") {
      return phoenix_tool::RunFeatureMatcher(argc - 1, argv + 1);
    }

    if (command == "reconstruction" || command == "mapper") {
      return phoenix_tool::RunCommand(
          "colmap", phoenix_tool::SliceArgs(argc, argv, 2), "mapper");
    }

    if (command == "automatic_reconstructor") {
      return phoenix_tool::RunAutomaticReconstructor(
          phoenix_tool::ParseAutomaticOptions(argc, argv));
    }

    phoenix_tool::PrintHelp();
    return EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}