# CV-Demo 项目

## 项目概述

这是一个计算机视觉演示项目，包含风格迁移、人脸跟踪和AR导出等功能模块。

## 项目结构


CV-Demo/
├── demo_env/            # 环境配置文件
│   └── environment.yml
├── doc/                 # 项目文档
│   └── Note.md
└── src/                 # 源代码
    ├── ar_export/       # AR导出模块
    │   ├── depth_img_make.py
    │   └── point_make.py
    ├── face_tracking/    # 人脸跟踪模块
    │   ├── face_camera/
    │   ├── face_img/
    │   └── face_iphone/
    └── style_transfer/  # 风格迁移模块
        └── transfer.py

##  环境配置

1. 安装conda环境：

conda env create -f demo_env/environment.yml

## 使用说明

1. 风格迁移模块：运行 `src/style_transfer/transfer.py`
2. 人脸跟踪模块：摄像头检测：`src/face_tracking/face_camera/`中的脚本

   图片检测：`src/face_tracking/face_img/`中的脚本
3. AR导出模块：深度图生成：`src/ar_export/depth_img_make.py`

   点云生成：`src/ar_export/point_make.py`

## 依赖库

本项目依赖已包含在conda环境配置中，主要包含：

OpenCV (用于图像处理和计算机视觉)

dlib (用于人脸检测和特征提取)

其他依赖详见 `demo_env/environment.yml`

所有模型文件已内置在 `src/models/`目录中，无需额外下载。
