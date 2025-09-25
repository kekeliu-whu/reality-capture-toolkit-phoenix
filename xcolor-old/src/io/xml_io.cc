#include "io/xml_io.h"

#include <proj.h>
#include <tinyxml.h>
#include <boost/format.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// 空间参考系统定义
struct SpatialReferenceSystem {
  int id;
  std::string name;
  std::string definition;
};

// 图像尺寸
struct ImageDimensions {
  int width;
  int height;
};

// 主点坐标
struct PrincipalPoint {
  double x;
  double y;
};

// 畸变参数
struct Distortion {
  double k1, k2, k3;  // 径向畸变系数
  double p1, p2;      // 切向畸变系数
};

// 3x3旋转矩阵
struct RotationMatrix {
  double m_00, m_01, m_02;
  double m_10, m_11, m_12;
  double m_20, m_21, m_22;
};

// 相机中心坐标
struct CameraCenter {
  double x, y, z;
};

// 相机姿态
struct Pose {
  RotationMatrix rotation;
  CameraCenter center;
};

// 照片信息
struct Photo {
  int id;
  std::string image_path;
  int component;
  double near_depth;
  double median_depth;
  double far_depth;
  Pose pose;
};

// 照片组
struct Photogroup {
  ImageDimensions image_dimensions;
  std::string camera_model_type;
  double focal_length_pixels;
  double sensor_size;
  double focal_length;
  std::string camera_orientation;
  PrincipalPoint principal_point;
  Distortion distortion;
  double aspect_ratio;
  std::vector<Photo> photos;
};

// 3D位置
struct Position3D {
  double x, y, z;
};

// RGB颜色
struct Color {
  double red, green, blue;
};

// 测量点（图像坐标）
struct Measurement {
  int photo_id;
  double x, y;
};

// 连接点
struct TiePoint {
  Position3D position;
  Color color;
  std::vector<Measurement> measurements;
};

// 块（Block）
struct Block {
  int srs_id;
  std::vector<Photogroup> photogroups;
  std::vector<TiePoint> tie_points;
};

// 主要的BlocksExchange类
class BlocksExchange {
 public:
  std::string file_version;
  std::vector<SpatialReferenceSystem> spatial_reference_systems;
  Block block;

  BlocksExchange() : file_version("3.2") {}

  // 保存到XML文件
  bool SaveToXML(const std::string& filename) const {
    TiXmlDocument doc;

    // XML declaration
    TiXmlDeclaration* decl = new TiXmlDeclaration("1.0", "UTF-8", "");
    doc.LinkEndChild(decl);

    // Root element
    TiXmlElement* root = new TiXmlElement("BlocksExchange");
    root->SetAttribute("version", file_version.c_str());
    doc.LinkEndChild(root);

    // SpatialReferenceSystems
    TiXmlElement* srsElement = new TiXmlElement("SpatialReferenceSystems");
    root->LinkEndChild(srsElement);

    for (const auto& srs : spatial_reference_systems) {
      TiXmlElement* srsNode = new TiXmlElement("SRS");
      srsElement->LinkEndChild(srsNode);

      AddTextElement(srsNode, "Id", std::to_string(srs.id));
      AddTextElement(srsNode, "Name", srs.name);
      AddTextElement(srsNode, "Definition", srs.definition);
    }

    // Block
    TiXmlElement* blockElement = new TiXmlElement("Block");
    root->LinkEndChild(blockElement);

    AddTextElement(blockElement, "SRSId", std::to_string(block.srs_id));

    // Photogroups
    TiXmlElement* photogroupsElement = new TiXmlElement("Photogroups");
    blockElement->LinkEndChild(photogroupsElement);

    for (const auto& pg : block.photogroups) {
      TiXmlElement* pgElement = new TiXmlElement("Photogroup");
      photogroupsElement->LinkEndChild(pgElement);

      // ImageDimensions
      TiXmlElement* dimElement = new TiXmlElement("ImageDimensions");
      pgElement->LinkEndChild(dimElement);
      AddTextElement(dimElement, "Width", std::to_string(pg.image_dimensions.width));
      AddTextElement(dimElement, "Height", std::to_string(pg.image_dimensions.height));

      AddTextElement(pgElement, "CameraModelType", pg.camera_model_type);
      AddTextElement(pgElement, "FocalLengthPixels", FormatDouble(pg.focal_length_pixels, 9));
      AddTextElement(pgElement, "SensorSize", FormatDouble(pg.sensor_size, 9));
      AddTextElement(pgElement, "FocalLength", FormatDouble(pg.focal_length, 9));
      AddTextElement(pgElement, "CameraOrientation", pg.camera_orientation);

      // PrincipalPoint
      TiXmlElement* ppElement = new TiXmlElement("PrincipalPoint");
      pgElement->LinkEndChild(ppElement);
      AddTextElement(ppElement, "x", FormatDouble(pg.principal_point.x, 9));
      AddTextElement(ppElement, "y", FormatDouble(pg.principal_point.y, 9));

      // Distortion
      TiXmlElement* distElement = new TiXmlElement("Distortion");
      pgElement->LinkEndChild(distElement);
      AddTextElement(distElement, "K1", FormatDouble(pg.distortion.k1, 9));
      AddTextElement(distElement, "K2", FormatDouble(pg.distortion.k2, 9));
      AddTextElement(distElement, "K3", FormatDouble(pg.distortion.k3, 9));
      AddTextElement(distElement, "P1", FormatDouble(pg.distortion.p1, 9));
      AddTextElement(distElement, "P2", FormatDouble(pg.distortion.p2, 9));

      AddTextElement(pgElement, "AspectRatio", FormatDouble(pg.aspect_ratio, 9));

      // Photos
      for (const auto& photo : pg.photos) {
        TiXmlElement* photoElement = new TiXmlElement("Photo");
        pgElement->LinkEndChild(photoElement);

        AddTextElement(photoElement, "Id", std::to_string(photo.id));
        AddTextElement(photoElement, "ImagePath", photo.image_path);
        AddTextElement(photoElement, "Component", std::to_string(photo.component));
        AddTextElement(photoElement, "NearDepth", FormatDouble(photo.near_depth, 9));
        AddTextElement(photoElement, "MedianDepth", FormatDouble(photo.median_depth, 9));
        AddTextElement(photoElement, "FarDepth", FormatDouble(photo.far_depth, 9));

        // Pose
        TiXmlElement* poseElement = new TiXmlElement("Pose");
        photoElement->LinkEndChild(poseElement);

        // Rotation
        TiXmlElement* rotElement = new TiXmlElement("Rotation");
        poseElement->LinkEndChild(rotElement);
        AddTextElement(rotElement, "M_00", FormatDouble(photo.pose.rotation.m_00, 9));
        AddTextElement(rotElement, "M_01", FormatDouble(photo.pose.rotation.m_01, 9));
        AddTextElement(rotElement, "M_02", FormatDouble(photo.pose.rotation.m_02, 9));
        AddTextElement(rotElement, "M_10", FormatDouble(photo.pose.rotation.m_10, 9));
        AddTextElement(rotElement, "M_11", FormatDouble(photo.pose.rotation.m_11, 9));
        AddTextElement(rotElement, "M_12", FormatDouble(photo.pose.rotation.m_12, 9));
        AddTextElement(rotElement, "M_20", FormatDouble(photo.pose.rotation.m_20, 9));
        AddTextElement(rotElement, "M_21", FormatDouble(photo.pose.rotation.m_21, 9));
        AddTextElement(rotElement, "M_22", FormatDouble(photo.pose.rotation.m_22, 9));

        // Center
        TiXmlElement* centerElement = new TiXmlElement("Center");
        poseElement->LinkEndChild(centerElement);
        AddTextElement(centerElement, "x", FormatDouble(photo.pose.center.x, 9));
        AddTextElement(centerElement, "y", FormatDouble(photo.pose.center.y, 9));
        AddTextElement(centerElement, "z", FormatDouble(photo.pose.center.z, 9));
      }
    }

    // TiePoints
    TiXmlElement* tiePointsElement = new TiXmlElement("TiePoints");
    blockElement->LinkEndChild(tiePointsElement);

    for (const auto& tp : block.tie_points) {
      TiXmlElement* tpElement = new TiXmlElement("TiePoint");
      tiePointsElement->LinkEndChild(tpElement);

      // Position
      TiXmlElement* posElement = new TiXmlElement("Position");
      tpElement->LinkEndChild(posElement);
      AddTextElement(posElement, "x", FormatDouble(tp.position.x, 9));
      AddTextElement(posElement, "y", FormatDouble(tp.position.y, 9));
      AddTextElement(posElement, "z", FormatDouble(tp.position.z, 9));

      // Color
      TiXmlElement* colorElement = new TiXmlElement("Color");
      tpElement->LinkEndChild(colorElement);
      AddTextElement(colorElement, "Red", FormatDouble(tp.color.red, 9));
      AddTextElement(colorElement, "Green", FormatDouble(tp.color.green, 9));
      AddTextElement(colorElement, "Blue", FormatDouble(tp.color.blue, 9));

      // Measurements
      for (const auto& meas : tp.measurements) {
        TiXmlElement* measElement = new TiXmlElement("Measurement");
        tpElement->LinkEndChild(measElement);

        AddTextElement(measElement, "PhotoId", std::to_string(meas.photo_id));
        AddTextElement(measElement, "x", FormatDouble(meas.x, 9));
        AddTextElement(measElement, "y", FormatDouble(meas.y, 9));
      }
    }

    return doc.SaveFile(filename.c_str());
  }

  // 添加空间参考系统
  void AddSpatialReferenceSystem(const SpatialReferenceSystem& srs) { spatial_reference_systems.push_back(srs); }

  // 添加照片组
  void AddPhotogroup(const Photogroup& pg) { block.photogroups.push_back(pg); }

  // 添加连接点
  void AddTiePoint(const TiePoint& tp) { block.tie_points.push_back(tp); }

 private:
  // Helper function to format double values with specified precision
  std::string FormatDouble(double value, int precision) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
  }

  // Helper function to add text elements
  void AddTextElement(TiXmlElement* parent, const std::string& name, const std::string& text) const {
    TiXmlElement* element = new TiXmlElement(name.c_str());
    TiXmlText* textNode   = new TiXmlText(text.c_str());
    element->LinkEndChild(textNode);
    parent->LinkEndChild(element);
  }
};

namespace {

Eigen::Vector2d GetLocalENU(const Eigen::Vector2d& coord, const Eigen::Vector2d& offset, const std::string& proj_str, const Eigen::Vector2d& origin) {
  Eigen::Vector2d coord_with_offset = coord + offset;

  // 初始化PROJ上下文
  PJ_CONTEXT* ctx = proj_context_create();

  // 构建源投影和目标ENU投影字符串
  std::ostringstream oss;
  oss << "+proj=aeqd +lat_0=" << std::fixed << std::setprecision(9) << origin.y() 
      << " +lon_0=" << std::fixed << std::setprecision(9) << origin.x() 
      << " +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
  std::string target_enu_proj_str = oss.str();

  // 直接创建从源投影到目标ENU的转换管道
  PJ* transformer = proj_create_crs_to_crs(ctx, proj_str.c_str(), target_enu_proj_str.c_str(), nullptr);

  if (!transformer) {
    proj_context_destroy(ctx);
    return Eigen::Vector2d::Zero();
  }

  // 执行直接转换
  PJ_COORD coord_source = proj_coord(coord_with_offset.x(), coord_with_offset.y(), 0, 0);
  PJ_COORD coord_enu    = proj_trans(transformer, PJ_FWD, coord_source);

  Eigen::Vector2d result = Eigen::Vector2d::Zero();
  if (coord_enu.xy.x != HUGE_VAL && coord_enu.xy.y != HUGE_VAL) {
    result.x() = coord_enu.xy.x;  // East
    result.y() = coord_enu.xy.y;  // North
  }

  // 清理资源
  proj_destroy(transformer);
  proj_context_destroy(ctx);

  return result;
}

}  // namespace

namespace xcolor {

void SaveXml(const std::string& filename, const std::unordered_map<colmap::image_t, colmap::Image>& images,
             const std::unordered_map<colmap::camera_t, colmap::Camera>& cameras, std::vector<MatchTrack>& match_tracks,
             const Eigen::Vector2d& offset, const std::string& proj_str, const std::string& images_path) {
  // todo kk
  double longitude = 113.4772024985;
  double latitude  = 22.8849155248;

  BlocksExchange be;

  SpatialReferenceSystem srs;
  srs.id         = 0;
  srs.name       = (boost::format("Local East-North-Up(ENU); origin: %.9fN %.9fE") % latitude % longitude).str();
  srs.definition = (boost::format("ENU:%.9f,%.9f") % latitude % longitude).str();
  be.AddSpatialReferenceSystem(srs);

  be.block.srs_id = srs.id;

  for (auto& [camera_id, camera] : cameras) {
    std::unordered_map<colmap::image_t, colmap::Image> select_images;
    for (auto& [image_id, image] : images) {
      if (image.CameraId() == camera_id) {
        select_images[image_id] = image;
      }
    }
    if (select_images.empty()) {
      continue;
    }

    Photogroup pg;
    pg.aspect_ratio            = 1.0;
    pg.camera_model_type       = "Perspective";
    pg.camera_orientation      = "XRightYDown";
    pg.distortion.k1           = camera.params[3];
    pg.distortion.k2           = camera.params[4];
    pg.distortion.p1           = camera.params[5];
    pg.distortion.p2           = camera.params[6];
    pg.distortion.k3           = 0;
    pg.principal_point.x       = camera.PrincipalPointX();
    pg.principal_point.y       = camera.PrincipalPointY();
    pg.sensor_size             = 24;
    pg.focal_length_pixels     = camera.FocalLength();
    pg.focal_length            = camera.FocalLength() / std::max(camera.width, camera.height) * pg.sensor_size;
    pg.image_dimensions.width  = camera.width;
    pg.image_dimensions.height = camera.height;
    for (auto& [image_id, image] : select_images) {
      Photo photo;
      photo.id                  = image_id;
      photo.far_depth           = 200;
      photo.near_depth          = 2;
      photo.median_depth        = 40;
      photo.component           = 1;
      photo.image_path          = images_path + "/" + image.Name();
      auto pos                  = colmap::Inverse(image.CamFromWorld()).translation;
      auto rot                  = image.CamFromWorld().rotation.matrix();
      Eigen::Vector2d local_enu = GetLocalENU(pos.head<2>(), offset, proj_str, Eigen::Vector2d(longitude, latitude));
      photo.pose.center.x       = local_enu.x();
      photo.pose.center.y       = local_enu.y();
      photo.pose.center.z       = pos.z();
      photo.pose.rotation.m_00  = rot(0, 0);
      photo.pose.rotation.m_01  = rot(0, 1);
      photo.pose.rotation.m_02  = rot(0, 2);
      photo.pose.rotation.m_10  = rot(1, 0);
      photo.pose.rotation.m_11  = rot(1, 1);
      photo.pose.rotation.m_12  = rot(1, 2);
      photo.pose.rotation.m_20  = rot(2, 0);
      photo.pose.rotation.m_21  = rot(2, 1);
      photo.pose.rotation.m_22  = rot(2, 2);
      pg.photos.push_back(photo);
    }
    be.block.photogroups.push_back(pg);
  }

  int dupPoint2D_count = 0;
  for (auto& track : match_tracks) {
    TiePoint tp;
    tp.color.blue             = 1;
    tp.color.green            = 1;
    tp.color.red              = 1;
    Eigen::Vector2d local_enu = GetLocalENU(track.point3D.point3D.head<2>(), offset, proj_str, Eigen::Vector2d(longitude, latitude));
    tp.position.x             = local_enu.x();
    tp.position.y             = local_enu.y();
    tp.position.z             = track.point3D.point3D.z();
    std::set<colmap::image_t> image_ids;
    for (auto& point2D_on_image : track.point2D_on_imageN) {
      if (image_ids.find(point2D_on_image->image_id) != image_ids.end()) {
        dupPoint2D_count++;
        continue;
      }
      image_ids.insert(point2D_on_image->image_id);

      Measurement mea;
      mea.photo_id = point2D_on_image->image_id;
      mea.x        = point2D_on_image->point_pixel.x();
      mea.y        = point2D_on_image->point_pixel.y();
      tp.measurements.push_back(mea);
    }
    be.AddTiePoint(tp);
  }
  LOG(WARNING) << "dupPoint2D_count: " << dupPoint2D_count;

  be.SaveToXML(filename);
}

}  // namespace xcolor
