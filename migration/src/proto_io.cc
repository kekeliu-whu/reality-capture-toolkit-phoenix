
#include <google/protobuf/util/json_util.h>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zstd.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "migration/proto_io.h"

void FastGzipString(const std::string &uncompressed, std::string *compressed) {
  boost::iostreams::filtering_ostream out;
  out.push(boost::iostreams::zstd_compressor(boost::iostreams::zstd::default_compression));
  out.push(boost::iostreams::back_inserter(*compressed));
  boost::iostreams::write(out, reinterpret_cast<const char *>(uncompressed.data()), uncompressed.size());
}

void FastGunzipString(const std::string &compressed, std::string *decompressed) {
  boost::iostreams::filtering_ostream out;
  out.push(boost::iostreams::zstd_decompressor());
  out.push(boost::iostreams::back_inserter(*decompressed));
  boost::iostreams::write(out, reinterpret_cast<const char *>(compressed.data()), compressed.size());
}

namespace {

bool ReadSingleMsgFile(const std::string &filename, google::protobuf::Message &message) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);

  if (!file.is_open()) {
    spdlog::info("Failed to open file: {}", filename);
    return false;
  }

  if (!message.ParseFromIstream(&file)) {
    spdlog::info("Failed to parse EncoderFile from file: {}", filename);
    return false;
  }

  return true;
}

bool WriteSingleMsgFile(const std::string &filename, const google::protobuf::Message &message) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);

  if (!outfile.is_open()) {
    spdlog::info("Failed to open file: {}", filename);
    return false;
  }

  if (!message.SerializeToOstream(&outfile)) {
    spdlog::info("Failed to serialize EncoderFile to file: {}", filename);
    return false;
  }

  return true;
}

}  // namespace

bool WriteDelimitedTo(const google::protobuf::MessageLite &message, std::ofstream &rawOutput) {
  std::string buffer, buffer_compressed;
  message.SerializeToString(&buffer);
  FastGzipString(buffer, &buffer_compressed);

  const int len = buffer_compressed.size();
  rawOutput.write((char *)&len, sizeof(len));
  rawOutput.write(buffer_compressed.data(), buffer_compressed.size());

  if (!rawOutput) {
    return false;
  }

  return true;
}

bool ReadDelimitedFrom(std::ifstream &rawInput, google::protobuf::MessageLite *message) {
  if (!rawInput) {
    return false;
  }

  int len = 0;
  rawInput.read((char *)&len, sizeof(len));

  if (len <= 0) {
    return false;
  }

  std::string buffer(len, 0);
  rawInput.read(&buffer[0], len);

  if (!rawInput) {
    return false;
  }

  std::string buffer_decompressed;
  FastGunzipString(buffer, &buffer_decompressed);
  return message->ParseFromString(buffer_decompressed);
}

bool WriteLidarFile(const std::string &filename, const std::vector<ConstPtr<proto::LidarMsg>> &scans) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);
  if (!outfile.is_open()) {
    spdlog::info("Failed to open file: {}", filename);
    return false;
  }

  for (const auto &scan : scans) {
    if (!WriteDelimitedTo(*scan, outfile)) {
      spdlog::info("Failed to serialize LidarScanMsg to file: {}", filename);
      return false;
    }
  }
  return true;
}

bool ReadUndistortedLidarFile(const std::string &filename, std::function<void(const ConstPtr<proto::UndistoredLidarMsg> &)> callback) {
  std::ifstream infile(filename, std::ios::in | std::ios::binary);
  if (!infile.is_open()) {
    spdlog::info("Failed to open file: {}", filename);
    return false;
  }

  Ptr<proto::UndistoredLidarMsg> scan{new proto::UndistoredLidarMsg()};
  while (ReadDelimitedFrom(infile, scan.get())) {
    callback(scan);
  }
  return true;
}

bool WriteUndistortedLidarFile(const std::string &filename, const std::vector<ConstPtr<proto::UndistoredLidarMsg>> &scans) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);
  if (!outfile.is_open()) {
    spdlog::info("Failed to open file: {}", filename);
    return false;
  }

  for (const auto &scan : scans) {
    if (!WriteDelimitedTo(*scan, outfile)) {
      spdlog::info("Failed to serialize proto::UndistoredLidarMsg to file: {}", filename);
      return false;
    }
  }
  return true;
}

bool ReadImuFile(const std::string &filename, proto::ImuMsgList &imu) { return ReadSingleMsgFile(filename, imu); }

bool WriteImuFile(const std::string &filename, const proto::ImuMsgList &imu) { return WriteSingleMsgFile(filename, imu); }

bool ReadEncoderFile(const std::string &filename, proto::EncoderMsgList &motor) { return ReadSingleMsgFile(filename, motor); }

bool WriteEncoderFile(const std::string &filename, const proto::EncoderMsgList &motor) { return WriteSingleMsgFile(filename, motor); }

bool ReadGnssFile(const std::string &filename, proto::GpsMsgList &gnss) { return ReadSingleMsgFile(filename, gnss); }

bool WriteGnssFile(const std::string &filename, const proto::GpsMsgList &gnss) { return WriteSingleMsgFile(filename, gnss); }

bool ReadPoseFile(const std::string &filename, proto::PoseMsgList &pose) { return ReadSingleMsgFile(filename, pose); }

bool WritePoseFile(const std::string &filename, const proto::PoseMsgList &pose) { return WriteSingleMsgFile(filename, pose); }

bool ReadPgoMetricsFile(const std::string &filename, proto::PgoMetrics &metrics) { return ReadSingleMsgFile(filename, metrics); }

bool WritePgoMetricsFile(const std::string &filename, const proto::PgoMetrics &metrics) { return WriteSingleMsgFile(filename, metrics); }

bool ReadSensorCalibrationFile(const std::string &filename, reality_capture::calibration::SensorCalibration &calibration) {
  return ReadSingleMsgFile(filename, calibration);
}

bool WriteSensorCalibrationFile(const std::string &filename,
                                const reality_capture::calibration::SensorCalibration &calibration) {
  return WriteSingleMsgFile(filename, calibration);
}

bool ReadPgoConfigFile(const std::string &filename, proto::PgoConfig &config) {
  // Find the file extension
  std::size_t ext_pos = filename.rfind('.');
  if (ext_pos == std::string::npos) {
    // No file extension found
    return false;
  }

  std::string extension = filename.substr(ext_pos);

  if (extension == ".dat") {
    // For .pb files, use ReadSingleMsgFile
    return ReadSingleMsgFile(filename, config);
  } else if (extension == ".json") {
    // For .json files, use a JSON parser
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
      return false;  // File not found
    }

    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string fileContent = ss.str();

    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = false;
    auto status                   = google::protobuf::util::JsonStringToMessage(fileContent, &config);

    if (!status.ok()) {
      spdlog::info("Failed to parse JSON: {}", status.ToString());
      return false;
    }

    return true;
  }

  spdlog::info("Unsupported file extension: {}", extension);
  return false;
}
