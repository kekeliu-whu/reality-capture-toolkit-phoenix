#include "migration/inc_las_writer.h"
#include "migration/string.h"

#include <cstdlib>
#include <iostream>
#include <pdal/Options.hpp>

namespace migration {

IncrementalLasWriter::IncrementalLasWriter() : m_writer(std::make_unique<IncrementalLasWriterBase>()), m_initialized(false) {}

IncrementalLasWriter::~IncrementalLasWriter() = default;

void IncrementalLasWriter::initialize(const std::string& filename, pdal::PointTable& table) {
  if (m_initialized) {
    spdlog::error("IncrementalLasWriter already initialized, cannot reinitialize");
    std::exit(EXIT_FAILURE);
  }

  if (!m_writer) {
    spdlog::error("IncrementalLasWriter internal writer is null");
    std::exit(EXIT_FAILURE);
  }

  try {
    // Configure output options
    pdal::Options options;
    options.add("filename", PlatformToUTF8(filename));
    options.add("scale_x", 1e-4);
    options.add("scale_y", 1e-4);
    options.add("scale_z", 1e-4);

    // Set options and prepare for writing
    m_writer->setOptions(options);
    m_writer->prepare(table);

    // Call ready() to complete initialization
    m_writer->ready(table);

    m_initialized = true;
    spdlog::info("IncrementalLasWriter initialized successfully with file: {}", filename);
  } catch (const std::exception& e) {
    spdlog::error("Failed to initialize IncrementalLasWriter: {}", e.what());
    std::exit(EXIT_FAILURE);
  }
}

void IncrementalLasWriter::writeView(const std::shared_ptr<pdal::PointView>& view) {
  if (!m_initialized) {
    spdlog::error("IncrementalLasWriter not initialized, call initialize() first");
    std::exit(EXIT_FAILURE);
  }

  if (!view) {
    spdlog::error("PointView cannot be null");
    std::exit(EXIT_FAILURE);
  }

  if (view->size() == 0) {
    spdlog::warn("PointView is empty, skipping write");
    return;
  }

  try {
    m_writer->write(view);
    spdlog::debug("Successfully wrote {} points", view->size());
  } catch (const std::exception& e) {
    spdlog::error("Failed to write PointView: {}", e.what());
    std::exit(EXIT_FAILURE);
  }
}

void IncrementalLasWriter::finalize(pdal::PointTable& table) {
  if (!m_initialized) {
    spdlog::error("IncrementalLasWriter not initialized");
    std::exit(EXIT_FAILURE);
  }

  try {
    // Update header and close file
    m_writer->done(table);
    m_initialized = false;
    spdlog::info("IncrementalLasWriter finalized successfully");
  } catch (const std::exception& e) {
    spdlog::error("Failed to finalize IncrementalLasWriter: {}", e.what());
    std::exit(EXIT_FAILURE);
  }
}

bool IncrementalLasWriter::isInitialized() const { return m_initialized; }

}  // namespace migration
