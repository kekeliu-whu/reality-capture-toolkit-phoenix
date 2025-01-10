
# Open source software dependencies
Here is a Markdown table listing the open source projects and their respective open source licenses:

| Project       | Open Source License |
|---------------|---------------------|
| PCL (Point Cloud Library) | [BSD License](https://github.com/PointCloudLibrary/pcl)  |w
| Sophus       | [MIT License](https://github.com/strasdat/Sophus)  |
| ulog_cpp     | [BSD License](https://github.com/PX4/ulog_cpp)     |
| Boost        | [Boost Software License](https://www.boost.org/)   |
| Eigen3       | [MPL2 License](https://gitlab.com/libeigen/eigen)  |
| yaml-cpp     | [MIT License](https://github.com/jbeder/yaml-cpp)  |
| libLAS       | [BSD License](https://github.com/libLAS/libLAS)    |
| Ceres Solver | [Apache License](https://github.com/ceres-solver/ceres-solver) |
| colmap       | [BSD License](https://github.com/colmap/colmap)    |


# Pipeline of processing
* Migration (optinal)
* SLAM
  * odometry
  * pgo/pgo (optinal)
  * pgo/point_cloud_process
* SFM
  * pgo/camera_pose_calib
  * sfm/colmap
* Render
  * xcolor
