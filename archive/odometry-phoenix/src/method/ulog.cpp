#include "method/ulog.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

#pragma pack(push, 1)

struct UlogFullState {
  uint64_t timestamp;
  float pos_x;
  float pos_y;
  float pos_z;
  float vel_x;
  float vel_y;
  float vel_z;
  float quat_w;
  float quat_x;
  float quat_y;
  float quat_z;
  float grav_x;
  float grav_y;
  float grav_z;
  float ba_x;
  float ba_y;
  float ba_z;
  float bg_x;
  float bg_y;
  float bg_z;
};

struct UlogIeskfAttribute {
  uint64_t timestamp;
  uint32_t sweep_id;
  float downsample_dis;

  float point_eig_x;
  float point_eig_y;
  float point_eig_z;
  float flat_ness;
  float norm_eig_x;
  float norm_eig_y;
  float norm_eig_z;
  float smooth_ness;
  uint32_t use_point_num;
  uint32_t total_point_num;
  float overlap_radio;

  uint32_t iter_num;
  float pos_tol_x;
  float pos_tol_y;
  float pos_tol_z;
  float rot_tol_x;
  float rot_tol_y;
  float rot_tol_z;
  float std_dev;
  double pos_std_x;
  double pos_std_y;
  double pos_std_z;
  double vel_std_x;
  double vel_std_y;
  double vel_std_z;
  double rot_std_x;
  double rot_std_y;
  double rot_std_z;
  double gravity_std_x;
  double gravity_std_y;
  double gravity_std_z;
  double acc_bias_std_x;
  double acc_bias_std_y;
  double acc_bias_std_z;
  double gyro_bias_std_x;
  double gyro_bias_std_y;
  double gyro_bias_std_z;

  double rot_update_x;
  double rot_update_y;
  double rot_update_z;
  double pos_update_x;
  double pos_update_y;
  double pos_update_z;
  double vel_update_x;
  double vel_update_y;
  double vel_update_z;
  double bg_update_x;
  double bg_update_y;
  double bg_update_z;
  double ba_update_x;
  double ba_update_y;
  double ba_update_z;
};

struct UlogIeskfStatePredict {
  uint64_t timestamp;
  uint32_t sweep_id;
  float pos_x;
  float pos_y;
  float pos_z;
  float vel_x;
  float vel_y;
  float vel_z;
  float quat_x;
  float quat_y;
  float quat_z;
  float quat_w;
};

struct UlogIeskfAttributePredict {
  uint64_t timestamp;
  uint32_t sweep_id;
  float dt;
  float gyro_true_x;
  float gyro_true_y;
  float gyro_true_z;
  float acc_true_x;
  float acc_true_y;
  float acc_true_z;
  float gyro_world_x;
  float gyro_world_y;
  float gyro_world_z;
  float acc_world_x;
  float acc_world_y;
  float acc_world_z;
};

struct UlogDebugMsg {
  uint64_t timestamp;
  double data[30];
};

#pragma pack(pop)

uint64_t TimestampMicroseconds(double timestamp) {
  return static_cast<uint64_t>(timestamp * 1e6);
}

}  // namespace

namespace middleware {

ULogStorage::~ULogStorage() {
  StopLog();
}

void ULogStorage::StartLog(const std::string& log_filename) {
  if (initialized_.load()) {
    return;
  }

  try {
    writer_ = std::make_shared<ulog_cpp::SimpleWriter>(log_filename, 0);

    writer_->writeMessageFormat(
        "FullState",
        std::vector<ulog_cpp::Field>{
            {"uint64_t", "timestamp"}, {"float", "pos_x"},
            {"float", "pos_y"},        {"float", "pos_z"},
            {"float", "vel_x"},        {"float", "vel_y"},
            {"float", "vel_z"},        {"float", "quat_w"},
            {"float", "quat_x"},       {"float", "quat_y"},
            {"float", "quat_z"},       {"float", "grav_x"},
            {"float", "grav_y"},       {"float", "grav_z"},
            {"float", "ba_x"},         {"float", "ba_y"},
            {"float", "ba_z"},         {"float", "bg_x"},
            {"float", "bg_y"},         {"float", "bg_z"},
        });

    writer_->writeMessageFormat(
        "IeskfAttribute",
        std::vector<ulog_cpp::Field>{
            {"uint64_t", "timestamp"},       {"uint32_t", "sweep_id"},
            {"float", "downsample_dis"},     {"float", "point_eig_x"},
            {"float", "point_eig_y"},        {"float", "point_eig_z"},
            {"float", "flat_ness"},          {"float", "norm_eig_x"},
            {"float", "norm_eig_y"},         {"float", "norm_eig_z"},
            {"float", "smooth_ness"},        {"uint32_t", "use_point_num"},
            {"uint32_t", "total_point_num"}, {"float", "overlap_radio"},
            {"uint32_t", "iter_num"},        {"float", "pos_tol_x"},
            {"float", "pos_tol_y"},          {"float", "pos_tol_z"},
            {"float", "rot_tol_x"},          {"float", "rot_tol_y"},
            {"float", "rot_tol_z"},          {"float", "std_dev"},
            {"double", "pos_std_x"},         {"double", "pos_std_y"},
            {"double", "pos_std_z"},         {"double", "vel_std_x"},
            {"double", "vel_std_y"},         {"double", "vel_std_z"},
            {"double", "rot_std_x"},         {"double", "rot_std_y"},
            {"double", "rot_std_z"},         {"double", "gravity_std_x"},
            {"double", "gravity_std_y"},     {"double", "gravity_std_z"},
            {"double", "acc_bias_std_x"},    {"double", "acc_bias_std_y"},
            {"double", "acc_bias_std_z"},    {"double", "gyro_bias_std_x"},
            {"double", "gyro_bias_std_y"},   {"double", "gyro_bias_std_z"},
            {"double", "rot_update_x"},      {"double", "rot_update_y"},
            {"double", "rot_update_z"},      {"double", "pos_update_x"},
            {"double", "pos_update_y"},      {"double", "pos_update_z"},
            {"double", "vel_update_x"},      {"double", "vel_update_y"},
            {"double", "vel_update_z"},      {"double", "bg_update_x"},
            {"double", "bg_update_y"},       {"double", "bg_update_z"},
            {"double", "ba_update_x"},       {"double", "ba_update_y"},
            {"double", "ba_update_z"},
        });

    writer_->writeMessageFormat(
        "IeskfStatePredict",
        std::vector<ulog_cpp::Field>{
            {"uint64_t", "timestamp"}, {"uint32_t", "sweep_id"},
            {"float", "pos_x"},        {"float", "pos_y"},
            {"float", "pos_z"},        {"float", "vel_x"},
            {"float", "vel_y"},        {"float", "vel_z"},
            {"float", "quat_x"},       {"float", "quat_y"},
            {"float", "quat_z"},       {"float", "quat_w"},
        });

    writer_->writeMessageFormat(
        "IeskfAttributePredict",
        std::vector<ulog_cpp::Field>{
            {"uint64_t", "timestamp"}, {"uint32_t", "sweep_id"},
            {"float", "dt"},           {"float", "gyro_true_x"},
            {"float", "gyro_true_y"},  {"float", "gyro_true_z"},
            {"float", "acc_true_x"},   {"float", "acc_true_y"},
            {"float", "acc_true_z"},   {"float", "gyro_world_x"},
            {"float", "gyro_world_y"}, {"float", "gyro_world_z"},
            {"float", "acc_world_x"},  {"float", "acc_world_y"},
            {"float", "acc_world_z"},
        });

    writer_->writeMessageFormat(
        "DebugMsgs",
        std::vector<ulog_cpp::Field>{
            {"uint64_t", "timestamp"}, {"double", "data0"},
            {"double", "data1"},       {"double", "data2"},
            {"double", "data3"},       {"double", "data4"},
            {"double", "data5"},       {"double", "data6"},
            {"double", "data7"},       {"double", "data8"},
            {"double", "data9"},       {"double", "data10"},
            {"double", "data11"},      {"double", "data12"},
            {"double", "data13"},      {"double", "data14"},
            {"double", "data15"},      {"double", "data16"},
            {"double", "data17"},      {"double", "data18"},
            {"double", "data19"},      {"double", "data20"},
            {"double", "data21"},      {"double", "data22"},
            {"double", "data23"},      {"double", "data24"},
            {"double", "data25"},      {"double", "data26"},
            {"double", "data27"},      {"double", "data28"},
            {"double", "data29"},
        });

    writer_->headerComplete();
    message_ids_[kFullState] = writer_->writeAddLoggedMessage("FullState");
    message_ids_[kIeskfAttribute] =
        writer_->writeAddLoggedMessage("IeskfAttribute");
    message_ids_[kIeskfStatePredict] =
        writer_->writeAddLoggedMessage("IeskfStatePredict");
    message_ids_[kIeskfAttributePredict] =
        writer_->writeAddLoggedMessage("IeskfAttributePredict");
    message_ids_[kDebugMsgs] = writer_->writeAddLoggedMessage("DebugMsgs");
    initialized_.store(true);
  } catch (const ulog_cpp::ExceptionBase& error) {
    writer_.reset();
    throw std::runtime_error("Failed to create ULog file " + log_filename +
                             ": " + error.what());
  }
}

void ULogStorage::StopLog() {
  if (!writer_) {
    initialized_.store(false);
    return;
  }
  if (initialized_.load()) {
    writer_->fsync();
  }
  writer_.reset();
  initialized_.store(false);
}

void ULogStorage::HandleFullState(double timestamp,
                                  const lixel::FullStateMsg& msg) {
  if (!initialized_.load()) {
    return;
  }
  UlogFullState data{};
  data.timestamp = TimestampMicroseconds(timestamp);
  data.pos_x = static_cast<float>(msg.p.x());
  data.pos_y = static_cast<float>(msg.p.y());
  data.pos_z = static_cast<float>(msg.p.z());
  data.vel_x = static_cast<float>(msg.v.x());
  data.vel_y = static_cast<float>(msg.v.y());
  data.vel_z = static_cast<float>(msg.v.z());
  data.quat_w = static_cast<float>(msg.q.w());
  data.quat_x = static_cast<float>(msg.q.x());
  data.quat_y = static_cast<float>(msg.q.y());
  data.quat_z = static_cast<float>(msg.q.z());
  data.grav_x = static_cast<float>(msg.gravity.x());
  data.grav_y = static_cast<float>(msg.gravity.y());
  data.grav_z = static_cast<float>(msg.gravity.z());
  data.ba_x = static_cast<float>(msg.ba.x());
  data.ba_y = static_cast<float>(msg.ba.y());
  data.ba_z = static_cast<float>(msg.ba.z());
  data.bg_x = static_cast<float>(msg.bg.x());
  data.bg_y = static_cast<float>(msg.bg.y());
  data.bg_z = static_cast<float>(msg.bg.z());
  writer_->writeData(message_ids_[kFullState], data);
}

void ULogStorage::HandleIeskfAttribute(
    double timestamp, const lixel::AttributeIESKF& msg) {
  if (!initialized_.load()) {
    return;
  }
  UlogIeskfAttribute data{};
  data.timestamp = TimestampMicroseconds(timestamp);
  data.sweep_id = msg.sweep_id;
  data.downsample_dis = msg.downsample_dis;
  data.point_eig_x = msg.jacobi.point_eig.x();
  data.point_eig_y = msg.jacobi.point_eig.y();
  data.point_eig_z = msg.jacobi.point_eig.z();
  data.flat_ness = msg.jacobi.flat_ness;
  data.norm_eig_x = msg.jacobi.norm_eig.x();
  data.norm_eig_y = msg.jacobi.norm_eig.y();
  data.norm_eig_z = msg.jacobi.norm_eig.z();
  data.smooth_ness = msg.jacobi.smooth_ness;
  data.use_point_num = msg.jacobi.use_point_num;
  data.total_point_num = msg.jacobi.total_point_num;
  data.overlap_radio = msg.jacobi.overlap_radio;
  data.iter_num = msg.iterate.iter_num;
  data.pos_tol_x = msg.iterate.pos_tol.x();
  data.pos_tol_y = msg.iterate.pos_tol.y();
  data.pos_tol_z = msg.iterate.pos_tol.z();
  data.rot_tol_x = msg.iterate.rot_tol.x();
  data.rot_tol_y = msg.iterate.rot_tol.y();
  data.rot_tol_z = msg.iterate.rot_tol.z();
  data.std_dev = msg.iterate.std_dev;
  data.pos_std_x = msg.iterate.pos_std.x();
  data.pos_std_y = msg.iterate.pos_std.y();
  data.pos_std_z = msg.iterate.pos_std.z();
  data.vel_std_x = msg.iterate.vel_std.x();
  data.vel_std_y = msg.iterate.vel_std.y();
  data.vel_std_z = msg.iterate.vel_std.z();
  data.rot_std_x = msg.iterate.rot_std.x();
  data.rot_std_y = msg.iterate.rot_std.y();
  data.rot_std_z = msg.iterate.rot_std.z();
  data.gravity_std_x = msg.iterate.gravity_std.x();
  data.gravity_std_y = msg.iterate.gravity_std.y();
  data.gravity_std_z = msg.iterate.gravity_std.z();
  data.acc_bias_std_x = msg.iterate.acc_bias_std.x();
  data.acc_bias_std_y = msg.iterate.acc_bias_std.y();
  data.acc_bias_std_z = msg.iterate.acc_bias_std.z();
  data.gyro_bias_std_x = msg.iterate.gyro_bias_std.x();
  data.gyro_bias_std_y = msg.iterate.gyro_bias_std.y();
  data.gyro_bias_std_z = msg.iterate.gyro_bias_std.z();
  data.rot_update_x = msg.iterate.rot_update.x();
  data.rot_update_y = msg.iterate.rot_update.y();
  data.rot_update_z = msg.iterate.rot_update.z();
  data.pos_update_x = msg.iterate.pos_update.x();
  data.pos_update_y = msg.iterate.pos_update.y();
  data.pos_update_z = msg.iterate.pos_update.z();
  data.vel_update_x = msg.iterate.vel_update.x();
  data.vel_update_y = msg.iterate.vel_update.y();
  data.vel_update_z = msg.iterate.vel_update.z();
  data.bg_update_x = msg.iterate.gyro_bias_update.x();
  data.bg_update_y = msg.iterate.gyro_bias_update.y();
  data.bg_update_z = msg.iterate.gyro_bias_update.z();
  data.ba_update_x = msg.iterate.acc_bias_update.x();
  data.ba_update_y = msg.iterate.acc_bias_update.y();
  data.ba_update_z = msg.iterate.acc_bias_update.z();
  writer_->writeData(message_ids_[kIeskfAttribute], data);
}

void ULogStorage::HandleIeskfStatePredict(
    int sweep_id, const lixel::StatePredict& msgs) {
  if (!initialized_.load()) {
    return;
  }
  for (const auto& msg : msgs) {
    UlogIeskfStatePredict data{};
    data.timestamp = TimestampMicroseconds(msg.timestamp);
    data.sweep_id = static_cast<uint32_t>(sweep_id);
    data.pos_x = msg.pos.x();
    data.pos_y = msg.pos.y();
    data.pos_z = msg.pos.z();
    data.vel_x = msg.vel.x();
    data.vel_y = msg.vel.y();
    data.vel_z = msg.vel.z();
    data.quat_x = msg.quat.x();
    data.quat_y = msg.quat.y();
    data.quat_z = msg.quat.z();
    data.quat_w = msg.quat.w();
    writer_->writeData(message_ids_[kIeskfStatePredict], data);
  }
}

void ULogStorage::HandleIeskfAttributePredict(
    int sweep_id, const lixel::AttributePredict& msgs) {
  if (!initialized_.load()) {
    return;
  }
  for (const auto& msg : msgs) {
    UlogIeskfAttributePredict data{};
    data.timestamp = TimestampMicroseconds(msg.timestamp);
    data.sweep_id = static_cast<uint32_t>(sweep_id);
    data.dt = msg.dt;
    data.gyro_true_x = msg.gyro_true.x();
    data.gyro_true_y = msg.gyro_true.y();
    data.gyro_true_z = msg.gyro_true.z();
    data.acc_true_x = msg.acc_true.x();
    data.acc_true_y = msg.acc_true.y();
    data.acc_true_z = msg.acc_true.z();
    data.gyro_world_x = msg.gyro_world.x();
    data.gyro_world_y = msg.gyro_world.y();
    data.gyro_world_z = msg.gyro_world.z();
    data.acc_world_x = msg.acc_world.x();
    data.acc_world_y = msg.acc_world.y();
    data.acc_world_z = msg.acc_world.z();
    writer_->writeData(message_ids_[kIeskfAttributePredict], data);
  }
}

void ULogStorage::HandleDebugMsgs(const lixel::DebugMsgs& msgs) {
  if (!initialized_.load()) {
    return;
  }
  for (const auto& msg : msgs) {
    UlogDebugMsg data{};
    data.timestamp = TimestampMicroseconds(msg.timestamp);
    std::memcpy(data.data, msg.data, sizeof(data.data));
    writer_->writeData(message_ids_[kDebugMsgs], data);
  }
}

}  // namespace middleware
