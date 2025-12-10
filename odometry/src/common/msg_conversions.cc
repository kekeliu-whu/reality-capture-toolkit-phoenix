
#include "migration/proto_io.h"

#include "msg_conversions.h"

Rigid3d FromProto(const proto::Rigid3d& proto) {
  return Rigid3d({proto.tx(), proto.ty(), proto.tz()}, {proto.rw(), proto.rx(), proto.ry(), proto.rz()});
}

Matrix3 FromProto(const proto::Matrix3& proto) {
  CHECK_EQ(proto.data().size(), 9);
  Matrix3 mat;
  for (int i = 0; i < 9; ++i) {
    mat(i / 3, i % 3) = proto.data(i);
  }
  return mat;
}

Vector3 FromProto(const proto::Vector3& proto) {
  CHECK_EQ(proto.data().size(), 3);
  Vector3 vec;
  for (int i = 0; i < 3; ++i) {
    vec(i) = proto.data(i);
  }
  return vec;
}

std::shared_ptr<ImuInstrinsic> FromProto(const proto::ImuInstrinsicTpmIcra2014& proto) {
  return std::make_shared<ImuInstrinsicTpmIcra2014>(FromProto(proto.ta()), FromProto(proto.ka()), FromProto(proto.ba()), FromProto(proto.tg()),
                                                    FromProto(proto.kg()), FromProto(proto.bg()));
}

std::shared_ptr<ImuInstrinsic> FromProto(const proto::ImuInstrinsic& proto) {
  if (proto.has_tpm()) {
    return FromProto(proto.tpm());
  } else {
    spdlog::critical("");
    exit(1);
  }
}

std::shared_ptr<LidarInstrinsic> FromProto(const proto::LidarInstrinsicSimple& proto) {
  CHECK_EQ(proto.s().size(), 2);
  LidarInstrinsicSimple simple;
  simple.elevation_corr = proto.e();
  simple.s[0]           = proto.s(0);
  simple.s[1]           = proto.s(1);
  return std::make_shared<LidarInstrinsicSimple>(simple);
}

std::shared_ptr<LidarInstrinsic> FromProto(const proto::LidarInstrinsic& proto) {
  if (proto.has_simple()) {
    return FromProto(proto.simple());
  } else {
    spdlog::critical("");
    exit(1);
  }
}

SensorCalib FromProto(const proto::SensorCalib& proto) {
  return SensorCalib(proto.has_encoder(), FromProto(proto.lidar_to_encoder()), FromProto(proto.encoder_to_imu()), FromProto(proto.lidar_instrinsic()),
                     FromProto(proto.imu_instrinsic()));
}

ImuMsg FromProto(const proto::ImuMsg& proto) {
  ImuMsg msg;
  msg.timestamp = proto.timestamp();
  msg.acc.x()   = proto.ax();
  msg.acc.y()   = proto.ay();
  msg.acc.z()   = proto.az();
  msg.gyr.x()   = proto.gx();
  msg.gyr.y()   = proto.gy();
  msg.gyr.z()   = proto.gz();
  return msg;
}

EncoderMsg FromProto(const proto::EncoderMsg& proto) {
  EncoderMsg msg;
  msg.timestamp = proto.timestamp();
  msg.rot.x()   = proto.rx();
  msg.rot.y()   = proto.ry();
  msg.rot.z()   = proto.rz();
  msg.rot.w()   = proto.rw();
  return msg;
}

ConstPtr<LidarMsg> FromProto(const ConstPtr<proto::LidarMsg>& proto) {
  Ptr<LidarMsg> msg(new LidarMsg);
  msg->lidar_points->reserve(proto->points_size());
  for (auto p : proto->points()) {
    PointXYZIRT point;
    point.x         = p.x();
    point.y         = p.y();
    point.z         = p.z();
    point.intensity = p.intensity();
    point.timestamp = p.timestamp();
    msg->lidar_points->push_back(point);
  }
  return msg;
}
