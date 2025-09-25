# XSFM

## 概述
一个点云辅助的SFM框架。


## 开发者指南
### 1 环境准备
#### 1.1 系统环境
* 操作系统：Windows 10及以上  
* 编译器：Visual Studio 2022及以上


#### 1.2 软件依赖
所有依赖都通过vcpkg安装，请按照如下步骤安装vcpkg：  
* 新建文件夹：手动建立`D:/library`文件夹，打开powershell，进入该文件夹
* 克隆代码：`git clone https://github.com/SaaS-Platform-GZPI/vcpkg`
* 进入vcpkg：`cd vcpkg`
* 安装vcpkg：`./bootstrap-vcpkg.bat`
* 环境变量：打开系统环境变量设置，在`系统变量-Path`中添加路径`D:/library/vcpkg`
* 重启系统：重启系统以让环境变量生效
* 验证安装：打开命令行，输入`vcpkg version`，如果显示版本号，则安装成功。

安装依赖软件，开发者可以按需安装自己需要的软件）：
`vcpkg install boost pcl colmap[cuda] opencv cgal pdal rapidjson gtest mimalloc[override] crashpad yaml-cpp --recurse`


#### 1.3 IDE配置
我们使用vscode和Visual Studio 2022作为IDE，您可以根据自己的需要选择其他IDE。
* 安装vscode
* 安装C/C++扩展
* 安装CMake Tools扩展


### 2 编译
* 使用vscode打开本项目
* 进入vscode的`CMake`面板，点击`Configure`按钮，选择`Visual Studio 2022 Release - amd64`作为编译器，选择`RelWithDebInfo`作为编译模式，等待CMake配置完成
* 重启vscode，让C/C++插件重新加载项目，这样编写代码的时候就可以使用智能提示了
* 点击左下方状态栏的`Build`按钮，等待编译完成，编译成功后算法服务和客户端示例程序会在`build/RelWithDebInfo`目录下生成，里面也包含了所有的依赖库


## 运行
### 特征提取和匹配
手动运行colmap，执行特征提取和匹配，得到xsfm.db，里面报错着sfm所需的匹配对。
### XSFM
执行XSFM进行POS优化。
