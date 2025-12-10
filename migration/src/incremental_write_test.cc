#include <gtest/gtest.h>
#include <mimalloc.h>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <pdal/PointTable.hpp>

#include "migration/inc_las_writer.h"

TEST(IncrementalLasWriterTest, BasicFunctionality) {
  spdlog::info("Starting incremental LAS write example...");

  // spdlog::info("malloc: {}", mi_version());

  // 1. Define point cloud data layout
  pdal::PointTable table;
  table.layout()->registerDim(pdal::Dimension::Id::X);
  table.layout()->registerDim(pdal::Dimension::Id::Y);
  table.layout()->registerDim(pdal::Dimension::Id::Z);
  table.layout()->registerDim(pdal::Dimension::Id::Intensity);

  // 2. Initialize writer
  migration::IncrementalLasWriter writer;
  writer.initialize("output_huge.las", table);
  spdlog::info("Writer initialized for file: output_huge.las");

  // 3. Simulate incremental write loop - buffer all chunks
  const size_t chunk_count      = 1000;
  const size_t points_per_chunk = 10000;

  for (size_t i = 0; i < chunk_count; ++i) {
    // Create point view for this chunk
    pdal::PointViewPtr view(new pdal::PointView(table));

    // Fill with sample data
    for (size_t j = 0; j < points_per_chunk; ++j) {
      double x = 1.0 * i + j * 0.01;
      double y = 2.0 * i + j * 0.01;
      double z = 3.0 * i + j * 0.01;

      view->setField(pdal::Dimension::Id::X, j, x);
      view->setField(pdal::Dimension::Id::Y, j, y);
      view->setField(pdal::Dimension::Id::Z, j, z);
      view->setField(pdal::Dimension::Id::Intensity, j, (uint16_t)i);
    }

    // Buffer this chunk
    writer.writeView(view);
    spdlog::info("Buffered chunk: {}/{}", i + 1, chunk_count);
  }

  // 4. Finalize - execute write operation once
  writer.finalize(table);
  spdlog::info("Write operation completed successfully!");
}
