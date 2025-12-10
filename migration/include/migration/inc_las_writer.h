#ifndef MIGRATION_INC_LAS_WRITER_H
#define MIGRATION_INC_LAS_WRITER_H

#include <pdal/pdal.hpp>
#include <pdal/PointView.hpp>
#include <pdal/io/LasWriter.hpp>
#include <spdlog/spdlog.h>
#include <memory>
#include <string>

namespace migration {

// Helper class: expose protected methods of PDAL LasWriter as public
class IncrementalLasWriterBase : public pdal::LasWriter {
public:
    using pdal::Writer::write;
    using pdal::Writer::done;
    using pdal::LasWriter::ready;
    using pdal::LasWriter::finishOutput;
};

/**
 * Incremental LAS file writer
 * For efficient writing of large-scale point cloud data, supporting batch
 * submission to control memory usage
 */
class IncrementalLasWriter {
public:
    IncrementalLasWriter();
    ~IncrementalLasWriter();

    /**
     * Initialize the writer
     * @param filename Output LAS file path
     * @param table Point cloud data table (defines layout and fields)
     */
    void initialize(const std::string& filename, pdal::PointTable& table);

    /**
     * Write a PointView chunk to file
     * @param view PointView object containing point data
     */
    void writeView(const std::shared_ptr<pdal::PointView>& view);

    /**
     * Finalize the write operation
     * Update LAS header information (total points, Min/Max bounding box, etc.)
     * and close the file
     * @param table Point cloud data table
     */
    void finalize(pdal::PointTable& table);

    /**
     * Check if writer is initialized
     */
    bool isInitialized() const;

private:
    std::unique_ptr<IncrementalLasWriterBase> m_writer;
    bool m_initialized;
};

} // namespace migration

#endif // MIGRATION_INC_LAS_WRITER_H
