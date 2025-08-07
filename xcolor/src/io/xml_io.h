#pragma once

#include <tinyxml.h>
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
  bool saveToXML(const std::string& filename) const {
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

      addTextElement(srsNode, "Id", std::to_string(srs.id));
      addTextElement(srsNode, "Name", srs.name);
      addTextElement(srsNode, "Definition", srs.definition);
    }

    // Block
    TiXmlElement* blockElement = new TiXmlElement("Block");
    root->LinkEndChild(blockElement);

    addTextElement(blockElement, "SRSId", std::to_string(block.srs_id));

    // Photogroups
    TiXmlElement* photogroupsElement = new TiXmlElement("Photogroups");
    blockElement->LinkEndChild(photogroupsElement);

    for (const auto& pg : block.photogroups) {
      TiXmlElement* pgElement = new TiXmlElement("Photogroup");
      photogroupsElement->LinkEndChild(pgElement);

      // ImageDimensions
      TiXmlElement* dimElement = new TiXmlElement("ImageDimensions");
      pgElement->LinkEndChild(dimElement);
      addTextElement(dimElement, "Width", std::to_string(pg.image_dimensions.width));
      addTextElement(dimElement, "Height", std::to_string(pg.image_dimensions.height));

      addTextElement(pgElement, "CameraModelType", pg.camera_model_type);
      addTextElement(pgElement, "FocalLengthPixels", formatDouble(pg.focal_length_pixels, 9));
      addTextElement(pgElement, "SensorSize", formatDouble(pg.sensor_size, 9));
      addTextElement(pgElement, "FocalLength", formatDouble(pg.focal_length, 9));
      addTextElement(pgElement, "CameraOrientation", pg.camera_orientation);

      // PrincipalPoint
      TiXmlElement* ppElement = new TiXmlElement("PrincipalPoint");
      pgElement->LinkEndChild(ppElement);
      addTextElement(ppElement, "x", formatDouble(pg.principal_point.x, 9));
      addTextElement(ppElement, "y", formatDouble(pg.principal_point.y, 9));

      // Distortion
      TiXmlElement* distElement = new TiXmlElement("Distortion");
      pgElement->LinkEndChild(distElement);
      addTextElement(distElement, "K1", formatDouble(pg.distortion.k1, 9));
      addTextElement(distElement, "K2", formatDouble(pg.distortion.k2, 9));
      addTextElement(distElement, "K3", formatDouble(pg.distortion.k3, 9));
      addTextElement(distElement, "P1", formatDouble(pg.distortion.p1, 9));
      addTextElement(distElement, "P2", formatDouble(pg.distortion.p2, 9));

      addTextElement(pgElement, "AspectRatio", formatDouble(pg.aspect_ratio, 9));

      // Photos
      for (const auto& photo : pg.photos) {
        TiXmlElement* photoElement = new TiXmlElement("Photo");
        pgElement->LinkEndChild(photoElement);

        addTextElement(photoElement, "Id", std::to_string(photo.id));
        addTextElement(photoElement, "ImagePath", photo.image_path);
        addTextElement(photoElement, "Component", std::to_string(photo.component));
        addTextElement(photoElement, "NearDepth", formatDouble(photo.near_depth, 9));
        addTextElement(photoElement, "MedianDepth", formatDouble(photo.median_depth, 9));
        addTextElement(photoElement, "FarDepth", formatDouble(photo.far_depth, 9));

        // Pose
        TiXmlElement* poseElement = new TiXmlElement("Pose");
        photoElement->LinkEndChild(poseElement);

        // Rotation
        TiXmlElement* rotElement = new TiXmlElement("Rotation");
        poseElement->LinkEndChild(rotElement);
        addTextElement(rotElement, "M_00", formatDouble(photo.pose.rotation.m_00, 9));
        addTextElement(rotElement, "M_01", formatDouble(photo.pose.rotation.m_01, 9));
        addTextElement(rotElement, "M_02", formatDouble(photo.pose.rotation.m_02, 9));
        addTextElement(rotElement, "M_10", formatDouble(photo.pose.rotation.m_10, 9));
        addTextElement(rotElement, "M_11", formatDouble(photo.pose.rotation.m_11, 9));
        addTextElement(rotElement, "M_12", formatDouble(photo.pose.rotation.m_12, 9));
        addTextElement(rotElement, "M_20", formatDouble(photo.pose.rotation.m_20, 9));
        addTextElement(rotElement, "M_21", formatDouble(photo.pose.rotation.m_21, 9));
        addTextElement(rotElement, "M_22", formatDouble(photo.pose.rotation.m_22, 9));

        // Center
        TiXmlElement* centerElement = new TiXmlElement("Center");
        poseElement->LinkEndChild(centerElement);
        addTextElement(centerElement, "x", formatDouble(photo.pose.center.x, 9));
        addTextElement(centerElement, "y", formatDouble(photo.pose.center.y, 9));
        addTextElement(centerElement, "z", formatDouble(photo.pose.center.z, 9));
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
      addTextElement(posElement, "x", formatDouble(tp.position.x, 9));
      addTextElement(posElement, "y", formatDouble(tp.position.y, 9));
      addTextElement(posElement, "z", formatDouble(tp.position.z, 9));

      // Color
      TiXmlElement* colorElement = new TiXmlElement("Color");
      tpElement->LinkEndChild(colorElement);
      addTextElement(colorElement, "Red", formatDouble(tp.color.red, 9));
      addTextElement(colorElement, "Green", formatDouble(tp.color.green, 9));
      addTextElement(colorElement, "Blue", formatDouble(tp.color.blue, 9));

      // Measurements
      for (const auto& meas : tp.measurements) {
        TiXmlElement* measElement = new TiXmlElement("Measurement");
        tpElement->LinkEndChild(measElement);

        addTextElement(measElement, "PhotoId", std::to_string(meas.photo_id));
        addTextElement(measElement, "x", formatDouble(meas.x, 9));
        addTextElement(measElement, "y", formatDouble(meas.y, 9));
      }
    }

    return doc.SaveFile(filename.c_str());
  }

  // 添加空间参考系统
  void addSpatialReferenceSystem(const SpatialReferenceSystem& srs) { spatial_reference_systems.push_back(srs); }

  // 添加照片组
  void addPhotogroup(const Photogroup& pg) { block.photogroups.push_back(pg); }

  // 添加连接点
  void addTiePoint(const TiePoint& tp) { block.tie_points.push_back(tp); }

 private:
  // Helper function to format double values with specified precision
  std::string formatDouble(double value, int precision) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
  }

  // Helper function to add text elements
  void addTextElement(TiXmlElement* parent, const std::string& name, const std::string& text) const {
    TiXmlElement* element = new TiXmlElement(name.c_str());
    TiXmlText* textNode   = new TiXmlText(text.c_str());
    element->LinkEndChild(textNode);
    parent->LinkEndChild(element);
  }
};
