# XMap

## XMap: a Hierarchical Mapping Method for XGRIDS 3D Reconstruction

## Introduction

**[XMap]** is a promotion about KD-Tree and IVOX with many new
features https://pecivkvtit.feishu.cn/drive/folder/ExzwfBV10l0NGOdgcR0cGe47nic

1. Efficient and Accuracy KNN Search based on Hierarchical Search
2. Lazy Update about KD-Tree, Reduce 10% Recall to Accelerate Map Incremental
3. Block Forget about Old Voxel for Local LIO
4. Normal Filter about Double-sided Wall for Indoor Scene
5. Dynamic Save and Load Map PointCloud in Disk for Global Map

### Developers:

[Yuan You 游远](https://github.com/SunnysTaste)

### Related paper

Related paper available on **arxiv**:

1. [FAST-LIO2: Fast Direct LiDAR-inertial Odometry](https://arxiv.org/abs/2107.06829)
2. [Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels](https://github.com/gaoxiang12/faster-lio/blob/main/doc/faster-lio.pdf)
3. [Voxelmap++: Mergeable Voxel Mapping Method for Online LiDAR(-inertial) Odometry](https://arxiv.org/pdf/2308.02799.pdf)
4. [Efficient and Probabilistic Adaptive Voxel Mapping for Accurate Online LiDAR Odometry](https://arxiv.org/abs/2109.07082)

## 1. Prerequisites

PCL >= 1.8, Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

Eigen >= 3.3.4, Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).

Yaml-Cpp >= 0.6.0, Follow [Yaml-Cpp Installation](https://github.com/jbeder/yaml-cpp).

## 2. Unittest

https://pecivkvtit.feishu.cn/drive/folder/PNqBf4drglSY8wdptMtcJhUlnUb

### 2.1 Extra Prerequisites

libLAS >= , Follow [libLAS Installation](http://www.pointclouds.org/downloads/linux.html).

GTEST >= , Follow [GTEST Installation](http://www.pointclouds.org/downloads/linux.html).

### 2.2 Build

Clone the repository and catkin_make:

```
    mkdir -p ~/xmap && cd ~/xmap
    git clone https://github.com/uestc-icsp/VoxelMapPlus_Public.git
    mkdir -p ~/xmap/build && cd ~/xmap/build
    cmake.. && make -j12
    ./incremental_test
```

### 2.3 Run

![incremental.png](picture/incremental.png)

## 3. Offline Running Results

https://pecivkvtit.feishu.cn/drive/folder/RchDfVt0RlGa7BduSlLcRPmznXe?from=space_personal_filelist

RedLine is the trajectory of LocalLIO, XMap have better accuracy compared with Xivt.
<div style="display:flex; justify-content:space-between;">
    <div>
        <h3>XMap LocalLIO</h3>
        <img src="picture/xmap_img.png" style="width:100%;">
    </div>
    <div style="width: 5%;"></div>
    <div>
        <h3>Xivt LocalLIO</h3>
        <img src="picture/xivt_img.png" style="width:82%;">
    </div>
</div>