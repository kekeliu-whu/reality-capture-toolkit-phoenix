
#include <fcntl.h>

#include "lio_core_ros/ulog.h"
#include "log/lsLogger.h"

namespace
{
#pragma pack(1)

struct UlogImu
{
  uint64_t timestamp;
  uint64_t collect_timestamp;
  float accel_x;
  float accel_y;
  float accel_z;
  float gryo_x;
  float gryo_y;
  float gryo_z;
  float temp;
};

struct UlogFullState
{
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

struct UlogIeskfAttribute
{
  uint64_t timestamp;
  uint32_t sweep_id;
  float downsample_dis;

  // AttributeJacobi
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

  // AttributeIterate
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

struct UlogIeskfStatePredict
{
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

struct UlogIeskfAttributePredict
{
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

struct UlogDebugMsg
{
  uint64_t timestamp;
  double data[30];
};

#pragma pack()
}  // namespace

namespace middleware
{

ULogStorage::ULogStorage()
{
}

ULogStorage::~ULogStorage()
{
  stopLog();
}

void ULogStorage::stopLog()
{
  if (!m_ulogger)
  {
    if (m_init_done)
    {
      m_ulogger->fsync();
    }
    m_ulogger.reset();
  }
}

void ULogStorage::startLog(std::string log_filename)
{
  if (m_init_done.load())
  {
    return;
  }

  try
  {
    m_ulogger = std::make_shared<ulog_cpp::SimpleWriter>(log_filename, 0);
  }
  catch (const ulog_cpp::ParsingException &e)
  {
    lslog(LSLOG_ERROR) << "Create ULogger failed, reason: " << e.what();
    m_ulogger.reset();
  }

  m_ulogger->writeInfo("sys_name", "ULogExampleWriter");

  m_ulogger->writeParameter("PARAM_A", 382.23F);
  m_ulogger->writeParameter("PARAM_B", 8272);

  m_ulogger->writeMessageFormat(
      "FullState",
      std::vector<ulog_cpp::Field>{
          {"uint64_t", "timestamp"}, {"float", "pos_x"},  {"float", "pos_y"},  {"float", "pos_z"},  {"float", "vel_x"},
          {"float", "vel_y"},        {"float", "vel_z"},  {"float", "quat_w"}, {"float", "quat_x"}, {"float", "quat_y"},
          {"float", "quat_z"},       {"float", "grav_x"}, {"float", "grav_y"}, {"float", "grav_z"}, {"float", "ba_x"},
          {"float", "ba_y"},         {"float", "ba_z"},   {"float", "bg_x"},   {"float", "bg_y"},   {"float", "bg_z"},
      });

  m_ulogger->writeMessageFormat(
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

  m_ulogger->writeMessageFormat(
      "IeskfStatePredict",
      std::vector<ulog_cpp::Field>{
          {"uint64_t", "timestamp"},
          {"uint32_t", "sweep_id"},
          {"float", "pos_x"},
          {"float", "pos_y"},
          {"float", "pos_z"},
          {"float", "vel_x"},
          {"float", "vel_y"},
          {"float", "vel_z"},
          {"float", "quat_x"},
          {"float", "quat_y"},
          {"float", "quat_z"},
          {"float", "quat_w"},
      });

  m_ulogger->writeMessageFormat(
      "IeskfAttributePredict",
      std::vector<ulog_cpp::Field>{
          {"uint64_t", "timestamp"},
          {"uint32_t", "sweep_id"},
          {"float", "dt"},
          {"float", "gyro_true_x"},
          {"float", "gyro_true_y"},
          {"float", "gyro_true_z"},
          {"float", "acc_true_x"},
          {"float", "acc_true_y"},
          {"float", "acc_true_z"},
          {"float", "gyro_world_x"},
          {"float", "gyro_world_y"},
          {"float", "gyro_world_z"},
          {"float", "acc_world_x"},
          {"float", "acc_world_y"},
          {"float", "acc_world_z"},
      });

  m_ulogger->writeMessageFormat(
      "DebugMsgs",
      std::vector<ulog_cpp::Field>{
          {"uint64_t", "timestamp"}, {"double", "data0"},  {"double", "data1"},  {"double", "data2"},
          {"double", "data3"},       {"double", "data4"},  {"double", "data5"},  {"double", "data6"},
          {"double", "data7"},       {"double", "data8"},  {"double", "data9"},  {"double", "data10"},
          {"double", "data11"},      {"double", "data12"}, {"double", "data13"}, {"double", "data14"},
          {"double", "data15"},      {"double", "data16"}, {"double", "data17"}, {"double", "data18"},
          {"double", "data19"},      {"double", "data20"}, {"double", "data21"}, {"double", "data22"},
          {"double", "data23"},      {"double", "data24"}, {"double", "data25"}, {"double", "data26"},
          {"double", "data27"},      {"double", "data28"}, {"double", "data29"},
      });

  m_ulogger->headerComplete();

  m_ulogIds[ULG_FullState] = m_ulogger->writeAddLoggedMessage("FullState");
  m_ulogIds[ULG_IeskfAttribute] = m_ulogger->writeAddLoggedMessage("IeskfAttribute");
  m_ulogIds[ULG_IeskfStatePredict] = m_ulogger->writeAddLoggedMessage("IeskfStatePredict");
  m_ulogIds[ULG_IeskfAttributePredict] = m_ulogger->writeAddLoggedMessage("IeskfAttributePredict");
  m_ulogIds[ULG_DebugMsgs] = m_ulogger->writeAddLoggedMessage("DebugMsgs");

  m_init_done.store(true);
}

void ULogStorage::HandleFullState(const lixel::FullStateMsg &msg)
{
  if (!m_init_done)
  {
    return;
  }
  struct UlogFullState log_full_state;
  log_full_state.timestamp = msg.timestamp * 1e6;
  log_full_state.pos_x = msg.p.x();
  log_full_state.pos_y = msg.p.y();
  log_full_state.pos_z = msg.p.z();
  log_full_state.vel_x = msg.v.x();
  log_full_state.vel_y = msg.v.y();
  log_full_state.vel_z = msg.v.z();

  log_full_state.quat_w = msg.q.w();
  log_full_state.quat_x = msg.q.x();
  log_full_state.quat_y = msg.q.y();
  log_full_state.quat_z = msg.q.z();
  // Eigen::Quaterniond q(msg.q.w(), msg.q.x(), msg.q.y(), msg.q.z());
  // Eigen::Matrix3d rotation_matrix = q.toRotationMatrix();
  // Eigen::AngleAxisd angle_axis(rotation_matrix);
  // Eigen::Vector3d axis_vector = angle_axis.axis() * angle_axis.angle();

  log_full_state.grav_x = msg.gravity.x();
  log_full_state.grav_y = msg.gravity.y();
  log_full_state.grav_z = msg.gravity.z();
  log_full_state.ba_x = msg.ba.x();
  log_full_state.ba_y = msg.ba.y();
  log_full_state.ba_z = msg.ba.z();
  log_full_state.bg_x = msg.bg.x();
  log_full_state.bg_y = msg.bg.y();
  log_full_state.bg_z = msg.bg.z();
  m_ulogger->writeData(m_ulogIds[ULG_FullState], log_full_state);
}

void ULogStorage::HandleIeskfAttribute(const lixel::AttributeIESKF &msg)
{
  if (!m_init_done)
  {
    return;
  }
  struct UlogIeskfAttribute ulog;
  ulog.timestamp = msg.timestamp * 1e6;
  ulog.sweep_id = msg.sweep_id;
  ulog.downsample_dis = msg.downsample_dis;

  ulog.point_eig_x = msg.jacobi.point_eig.x();
  ulog.point_eig_y = msg.jacobi.point_eig.y();
  ulog.point_eig_z = msg.jacobi.point_eig.z();
  ulog.flat_ness = msg.jacobi.flat_ness;
  ulog.norm_eig_x = msg.jacobi.norm_eig.x();
  ulog.norm_eig_y = msg.jacobi.norm_eig.y();
  ulog.norm_eig_z = msg.jacobi.norm_eig.z();
  ulog.smooth_ness = msg.jacobi.smooth_ness;
  ulog.use_point_num = msg.jacobi.use_point_num;
  ulog.total_point_num = msg.jacobi.total_point_num;
  ulog.overlap_radio = msg.jacobi.overlap_radio;

  ulog.iter_num = msg.iterate.iter_num;
  ulog.pos_tol_x = msg.iterate.pos_tol.x();
  ulog.pos_tol_y = msg.iterate.pos_tol.y();
  ulog.pos_tol_z = msg.iterate.pos_tol.z();
  ulog.rot_tol_x = msg.iterate.rot_tol.x();
  ulog.rot_tol_y = msg.iterate.rot_tol.y();
  ulog.rot_tol_z = msg.iterate.rot_tol.z();

  ulog.rot_update_x = msg.iterate.rot_update.x();
  ulog.rot_update_y = msg.iterate.rot_update.y();
  ulog.rot_update_z = msg.iterate.rot_update.z();
  ulog.pos_update_x = msg.iterate.pos_update.x();
  ulog.pos_update_y = msg.iterate.pos_update.y();
  ulog.pos_update_z = msg.iterate.pos_update.z();
  ulog.vel_update_x = msg.iterate.vel_update.x();
  ulog.vel_update_y = msg.iterate.vel_update.y();
  ulog.vel_update_z = msg.iterate.vel_update.z();
  ulog.bg_update_x = msg.iterate.gyro_bias_update.x();
  ulog.bg_update_y = msg.iterate.gyro_bias_update.y();
  ulog.bg_update_z = msg.iterate.gyro_bias_update.z();
  ulog.ba_update_x = msg.iterate.acc_bias_update.x();
  ulog.ba_update_y = msg.iterate.acc_bias_update.y();
  ulog.ba_update_z = msg.iterate.acc_bias_update.z();

  ulog.std_dev = msg.iterate.std_dev;
  ulog.pos_std_x = msg.iterate.pos_std.x();
  ulog.pos_std_y = msg.iterate.pos_std.y();
  ulog.pos_std_z = msg.iterate.pos_std.z();
  ulog.vel_std_x = msg.iterate.vel_std.x();
  ulog.vel_std_y = msg.iterate.vel_std.y();
  ulog.vel_std_z = msg.iterate.vel_std.z();
  ulog.rot_std_x = msg.iterate.rot_std.x();
  ulog.rot_std_y = msg.iterate.rot_std.y();
  ulog.rot_std_z = msg.iterate.rot_std.z();
  ulog.gravity_std_x = msg.iterate.gravity_std.x();
  ulog.gravity_std_y = msg.iterate.gravity_std.y();
  ulog.gravity_std_z = msg.iterate.gravity_std.z();
  ulog.acc_bias_std_x = msg.iterate.acc_bias_std.x();
  ulog.acc_bias_std_y = msg.iterate.acc_bias_std.y();
  ulog.acc_bias_std_z = msg.iterate.acc_bias_std.z();
  ulog.gyro_bias_std_x = msg.iterate.gyro_bias_std.x();
  ulog.gyro_bias_std_y = msg.iterate.gyro_bias_std.y();
  ulog.gyro_bias_std_z = msg.iterate.gyro_bias_std.z();
  m_ulogger->writeData(m_ulogIds[ULG_IeskfAttribute], ulog);
}

void ULogStorage::HandleIeskfStatePredict(int sweep_id, const lixel::StatePredict &msgs)
{
  if (!m_init_done)
  {
    return;
  }
  for (const lixel::PosAtt &msg : msgs)
  {
    struct UlogIeskfStatePredict ulog;
    ulog.timestamp = msg.timestamp * 1e6;
    ulog.sweep_id = sweep_id;
    ulog.pos_x = msg.pos.x();
    ulog.pos_y = msg.pos.y();
    ulog.pos_z = msg.pos.z();
    ulog.vel_x = msg.vel.x();
    ulog.vel_y = msg.vel.y();
    ulog.vel_z = msg.vel.z();
    ulog.quat_x = msg.quat.x();
    ulog.quat_y = msg.quat.y();
    ulog.quat_z = msg.quat.z();
    ulog.quat_w = msg.quat.w();
    m_ulogger->writeData(m_ulogIds[ULG_IeskfStatePredict], ulog);
  }
}

void ULogStorage::HandleIeskfAttributePredict(int sweep_id, const lixel::AttributePredict &msgs)
{
  if (!m_init_done)
  {
    return;
  }
  for (const lixel::AttributeImu &msg : msgs)
  {
    struct UlogIeskfAttributePredict ulog;
    ulog.timestamp = msg.timestamp * 1e6;
    ulog.sweep_id = sweep_id;
    ulog.dt = msg.dt;
    ulog.gyro_true_x = msg.gyro_true.x();
    ulog.gyro_true_y = msg.gyro_true.y();
    ulog.gyro_true_z = msg.gyro_true.z();

    ulog.acc_true_x = msg.acc_true.x();
    ulog.acc_true_y = msg.acc_true.y();
    ulog.acc_true_z = msg.acc_true.z();

    ulog.gyro_world_x = msg.gyro_world.x();
    ulog.gyro_world_y = msg.gyro_world.y();
    ulog.gyro_world_z = msg.gyro_world.z();

    ulog.acc_world_x = msg.acc_world.x();
    ulog.acc_world_y = msg.acc_world.y();
    ulog.acc_world_z = msg.acc_world.z();
    m_ulogger->writeData(m_ulogIds[ULG_IeskfAttributePredict], ulog);
  }
}

void ULogStorage::HandleDebugMsgs(const lixel::DebugMsgs &msgs)
{
  if (!m_init_done)
  {
    return;
  }
  for (const lixel::DebugMsg &msg : msgs)
  {
    struct UlogDebugMsg ulog;
    ulog.timestamp = msg.timestamp * 1e6;
    // DCHECK(sizeof(msg.data) == sizeof(ulog.data));
    memcpy(ulog.data, msg.data, sizeof(ulog.data));
    m_ulogger->writeData(m_ulogIds[ULG_DebugMsgs], ulog);
  }
}

}  // namespace middleware
