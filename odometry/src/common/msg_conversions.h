#pragma once

#include "migration/proto_io.h"
#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"
#include "types.h"

Rigid3d FromProto(const proto::Rigid3d& proto);

Matrix3 FromProto(const proto::Matrix3& proto);

Vector3 FromProto(const proto::Vector3& proto);

std::shared_ptr<ImuInstrinsic> FromProto(const proto::ImuInstrinsicTpmIcra2014& proto);

std::shared_ptr<ImuInstrinsic> FromProto(const proto::ImuInstrinsic& proto);

std::shared_ptr<LidarInstrinsic> FromProto(const proto::LidarInstrinsicSimple& proto);

std::shared_ptr<LidarInstrinsic> FromProto(const proto::LidarInstrinsic& proto);

SensorCalib FromProto(const proto::SensorCalib& proto);

ImuMsg FromProto(const proto::ImuMsg& proto);

EncoderMsg FromProto(const proto::EncoderMsg& proto);

ConstPtr<LidarMsg> FromProto(const ConstPtr<proto::LidarMsg>& proto);
