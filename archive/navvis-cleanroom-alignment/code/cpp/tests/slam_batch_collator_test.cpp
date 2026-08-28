#include "navvis_recon/slam_batch_collator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: slam_batch_collator_test ARCHIVE [BATCH_LIMIT]\n";
      return EXIT_FAILURE;
    }
    const std::size_t limit =
        argc == 3 ? static_cast<std::size_t>(std::stoull(argv[2])) : 100U;
    navvis_recon::slam::SlamBatchCollator collator(argv[1]);
    std::size_t points = 0;
    std::int64_t last_batch_timestamp = 0;
    for (std::size_t index = 0;
         index < limit && collator.hasNext(); ++index) {
      const navvis_recon::slam::TimedRangeBatch batch = collator.next();
      if (batch.points.empty() || batch.points.size() != batch.origins.size() ||
          batch.points.size() != batch.point_timestamps_ns.size() ||
          !std::is_sorted(batch.point_timestamps_ns.begin(),
                          batch.point_timestamps_ns.end()) ||
          batch.point_timestamps_ns.back() > batch.timestamp_ns ||
          (index != 0U && batch.timestamp_ns <= last_batch_timestamp)) {
        throw std::runtime_error("invalid collated SLAM batch");
      }
      points += batch.points.size();
      last_batch_timestamp = batch.timestamp_ns;
      std::cout << index << ',' << batch.timestamp_ns << ','
                << batch.points.size() << ','
                << batch.point_timestamps_ns.front() << ','
                << batch.point_timestamps_ns.back() << '\n';
    }
    std::cerr << "PASS total_batches=" << collator.batchCount()
              << " emitted=" << collator.nextBatchIndex()
              << " points=" << points << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
