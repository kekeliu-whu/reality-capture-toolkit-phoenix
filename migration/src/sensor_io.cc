
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zstd.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <cmath>
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <sstream>
#include <string>

#include "migration/sensor_io.h"

namespace {

void FastGzipString(const std::string &uncompressed, std::string *compressed) {
  boost::iostreams::filtering_ostream out;
  out.push(boost::iostreams::zstd_compressor(
      boost::iostreams::zstd::default_compression));
  out.push(boost::iostreams::back_inserter(*compressed));
  boost::iostreams::write(out,
                          reinterpret_cast<const char *>(uncompressed.data()),
                          uncompressed.size());
}

void FastGunzipString(const std::string &compressed,
                      std::string *decompressed) {
  boost::iostreams::filtering_ostream out;
  out.push(boost::iostreams::zstd_decompressor());
  out.push(boost::iostreams::back_inserter(*decompressed));
  boost::iostreams::write(out,
                          reinterpret_cast<const char *>(compressed.data()),
                          compressed.size());
}

bool WriteDelimitedTo(const google::protobuf::MessageLite &message,
                      std::ofstream &rawOutput) {

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

bool ReadDelimitedFrom(std::ifstream &rawInput,
                       google::protobuf::MessageLite *message) {
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

bool ReadSingleMsgFile(const std::string &filename,
                       google::protobuf::Message &message) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);

  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  if (!message.ParseFromIstream(&file)) {
    LOG(ERROR) << "Failed to parse MotorFile from file: " << filename;
    return false;
  }

  return true;
}

bool WriteSingleMsgFile(const std::string &filename,
                        const google::protobuf::Message &message) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);

  if (!outfile.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  if (!message.SerializeToOstream(&outfile)) {
    LOG(ERROR) << "Failed to serialize MotorFile to file: " << filename;
    return false;
  }

  return true;
}

} // namespace

bool ReadLidarFile(
    const std::string &filename,
    std::function<void(const std::shared_ptr<const LidarMsg> &)> callback) {
  std::ifstream infile(filename, std::ios::in | std::ios::binary);
  if (!infile.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  std::shared_ptr<LidarMsg> scan{new LidarMsg()};
  while (ReadDelimitedFrom(infile, scan.get())) {
    callback(scan);
  }
  return true;
}

bool WriteLidarFile(const std::string &filename,
                    const std::vector<ConstPtr<LidarMsg>> &scans) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);
  if (!outfile.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  for (const auto &scan : scans) {
    if (!WriteDelimitedTo(*scan, outfile)) {
      LOG(ERROR) << "Failed to serialize LidarScanMsg to file: " << filename;
      return false;
    }
  }
  return true;
}

bool ReadUndistortedLidarFile(
    const std::string &filename,
    std::function<void(const ConstPtr<UndistoredLidarMsg> &)> callback) {
  std::ifstream infile(filename, std::ios::in | std::ios::binary);
  if (!infile.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  Ptr<UndistoredLidarMsg> scan{new UndistoredLidarMsg()};
  while (ReadDelimitedFrom(infile, scan.get())) {
    callback(scan);
  }
  return true;
}

bool WriteUndistortedLidarFile(
    const std::string &filename,
    const std::vector<ConstPtr<UndistoredLidarMsg>> &scans) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);
  if (!outfile.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  for (const auto &scan : scans) {
    if (!WriteDelimitedTo(*scan, outfile)) {
      LOG(ERROR) << "Failed to serialize UndistoredLidarMsg to file: " << filename;
      return false;
    }
  }
  return true;
}

bool ReadImuFile(const std::string &filename, ImuMsgList &imu) {
  return ReadSingleMsgFile(filename, imu);
}

bool WriteImuFile(const std::string &filename, const ImuMsgList &imu) {
  return WriteSingleMsgFile(filename, imu);
}

bool ReadMotorFile(const std::string &filename, MotorMsgList &motor) {
  return ReadSingleMsgFile(filename, motor);
}

bool WriteMotorFile(const std::string &filename, const MotorMsgList &motor) {
  return WriteSingleMsgFile(filename, motor);
}

bool ReadPoseFile(const std::string &filename, PoseMsgList &pose) {
  return ReadSingleMsgFile(filename, pose);
}

bool WritePoseFile(const std::string &filename, const PoseMsgList &pose) {
  return WriteSingleMsgFile(filename, pose);
}

bool ReadPgoConfigFile(const std::string &filename, PgoConfig &config) {
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
      return false; // File not found
    }

    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string fileContent = ss.str();

    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = false;
    auto status =
        google::protobuf::util::JsonStringToMessage(fileContent, &config);

    if (!status.ok()) {
      LOG(ERROR) << "Failed to parse JSON: " << status.ToString();
      return false;
    }

    return true;
  }

  LOG(ERROR) << "Unsupported file extension: " << extension;
  return false;
}

bool LidarFileWriter::Write(const ConstPtr<UndistoredLidarMsg> &msg) {
  if (!WriteDelimitedTo(*msg, outfile_)) {
    LOG(ERROR) << "Failed to serialize UndistoredLidarMsg to file: "
               << filename_;
    return false;
  }
  return true;
}

bool LidarFileWriter::Open(const std::string &filename) {
  filename_ = filename;
  outfile_.open(filename, std::ios::out | std::ios::binary);
  LOG_IF(INFO, !outfile_.is_open()) << "Create lidar file: " << filename;
  return outfile_.is_open();
}

void LidarFileWriter::Close() { outfile_.close(); }

LidarFileWriter::~LidarFileWriter() { Close(); }

LidarFileWriter::LidarFileWriter() {}
