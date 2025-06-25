# YOLOv5 Ghost - 安全背心检测系统

<!-- ![YOLOv5](https://img.shields.io/badge/YOLOv5-v7.0-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red)
![License](https://img.shields.io/badge/License-AGPL--3.0-lightgrey) -->

基于 YOLOv5 Ghost 轻量化模型的安全背心检测系统，用于工业安全场景中的个人防护设备（PPE）检测。

## 📋 目录

- [项目简介](#项目简介)
- [功能特点](#功能特点)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [数据集](#数据集)
- [快速开始](#快速开始)
- [训练模型](#训练模型)
- [检测参数说明](#检测参数说明)
- [项目结构](#项目结构)
- [性能指标](#性能指标)
- [配置文件](#配置文件)
- [相关研究](#相关研究)
- [许可证](#许可证)

## 🎯 项目简介

本项目是基于 YOLOv5 架构的安全背心检测系统，专门用于识别工业环境中工人是否佩戴安全背心。该系统可以有效提高工业安全管理水平，降低工伤事故风险。

### 检测类别
- **NO-Safety Vest**: 未穿戴安全背心
- **Safety Vest**: 穿戴安全背心

## ✨ 功能特点

- 🚀 **高精度检测**: 基于 YOLOv5s 模型，检测精度高
- ⚡ **实时推理**: 支持实时视频流检测
- 🔧 **轻量化设计**: Ghost 模块优化，模型体积小
- 📊 **多种输出格式**: 支持图片、视频、摄像头实时检测
- 🎛️ **可配置参数**: 支持自定义置信度阈值、NMS 参数等
- 📱 **多平台部署**: 支持 CPU、GPU、移动端部署
- 🔄 **模型导出**: 支持 ONNX、TensorRT、CoreML 等格式

## 💻 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (GPU 训练推荐)
- 8GB+ RAM
- NVIDIA GPU (推荐，用于加速训练)

## 🛠️ 安装指南

### 方法一：使用 Conda 环境文件（推荐）

#### 1. 克隆项目

```bash
git clone https://github.com/LJK666666666/yolov5_ghost.git
cd yolov5_ghost
```

#### 2. 创建并激活 Conda 环境

```bash
# 使用 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate yolov5_ghost
```

#### 3. 验证安装

```bash
# 检查 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查主要依赖
python -c "import cv2, numpy, yaml, matplotlib; print('所有依赖安装成功！')"
```

### 方法二：手动安装

#### 1. 克隆项目

```bash
git clone https://github.com/LJK666666666/yolov5_ghost.git
cd yolov5_ghost
```

#### 2. 创建虚拟环境

```bash
# 使用 conda
conda create -n yolov5_ghost python=3.9
conda activate yolov5_ghost

# 或使用 venv
python -m venv yolov5_ghost
source yolov5_ghost/bin/activate  # Linux/Mac
# 或
yolov5_ghost\Scripts\activate  # Windows
```

#### 3. 安装 PyTorch

```bash
# CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU 版本 (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 下载预训练权重

```bash
# 下载 YOLOv5s 预训练权重
python -c "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt', 'yolov5s.pt')"
```

### 环境配置说明

- **GPU 用户**: 直接使用 `environment.yml`，已配置 CUDA 11.8 支持
- **CPU 用户**: 编辑 `environment.yml`，移除所有 CUDA 相关依赖，将 `pytorch-cuda=11.8` 改为 `cpuonly`
- **注意**: 本环境配置基于实际的 yolov5_ghost conda 环境导出

## 📊 数据集

本项目使用 Roboflow 安全背心数据集 v6：

- **训练集**: 包含多种工业场景的安全背心图像
- **验证集**: 用于模型验证和调优
- **测试集**: 用于最终性能评估
- **标注格式**: YOLO 格式 (txt 文件)
- **图像格式**: JPG/JPEG
- **数据来源**: [Roboflow Safety Vests Dataset](https://universe.roboflow.com/roboflow-universe-projects/safety-vests/dataset/6)

### 数据集结构

```
data/SafetyVests.v6/
├── train/
│   ├── images/     # 训练图像
│   └── labels/     # 训练标签
├── valid/
│   ├── images/     # 验证图像
│   └── labels/     # 验证标签
├── test/
│   ├── images/     # 测试图像
│   └── labels/     # 测试标签
└── data.yaml       # 数据集配置文件
```

## 🚀 快速开始

### 使用训练好的模型检测测试集

使用训练好的权重对 SafetyVests v6 测试数据集进行检测：

```bash
# 使用训练好的最佳权重检测测试集
python detect.py \
    --weights runs/train/exp3/weights/best.pt \
    --source data/SafetyVests.v6/test/images \
    --conf 0.25 \
    --save-txt \
    --save-conf \
    --project runs/test \
    --name safety_vest_test

# 检测单张测试图片
python detect.py \
    --weights runs/train/exp3/weights/best.pt \
    --source data/SafetyVests.v6/test/images/image_name.jpg \
    --conf 0.25 \
    --save-txt

# 批量检测并保存详细结果
python detect.py \
    --weights runs/train/exp3/weights/best.pt \
    --source data/SafetyVests.v6/test/images \
    --conf 0.25 \
    --iou-thres 0.45 \
    --save-txt \
    --save-conf \
    --save-crop \
    --line-thickness 2 \
    --project runs/detect \
    --name test_results
```

### 模型验证和评估

```bash
# 在验证集上评估模型性能
python val.py \
    --weights runs/train/exp3/weights/best.pt \
    --data data/SafetyVests.v6/data.yaml \
    --img 640 \
    --conf 0.001 \
    --iou 0.6 \
    --project runs/val \
    --name exp

# 在测试集上评估（如果测试集有标签）
python val.py \
    --weights runs/train/exp3/weights/best.pt \
    --data data/SafetyVests.v6/data.yaml \
    --task test \
    --img 640
```

## 🎓 训练模型

### 在 SafetyVests.v6 数据集上训练

```bash
# 基础训练命令
python train.py \
    --data data/SafetyVests.v6/data.yaml \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --batch-size 16 \
    --epochs 100 \
    --img-size 640 \
    --device 0

# 长期训练（更多轮次）
python train.py \
    --data data/SafetyVests.v6/data.yaml \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --batch-size 16 \
    --epochs 300 \
    --img-size 640 \
    --device 0 \
    --project runs/train \
    --name safety_vest_v6

# 恢复训练
python train.py --resume runs/train/exp3/weights/last.pt
```

### 训练监控

```bash
# 启动 TensorBoard 查看训练进度
tensorboard --logdir runs/train --port 6006
```

训练完成后，最佳模型权重将保存在 `runs/train/exp3/weights/best.pt`

## 🔍 检测参数说明

### 主要命令行参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--weights` | `yolov5s.pt` | 模型权重路径 |
| `--source` | `data/images` | 输入源路径 |
| `--img-size` | `640` | 推理图像大小 |
| `--conf-thres` | `0.25` | 置信度阈值 |
| `--iou-thres` | `0.45` | NMS IoU 阈值 |
| `--device` | `''` | 设备 (cpu, 0, 1, ...) |
| `--save-txt` | `False` | 保存结果到 txt |
| `--save-conf` | `False` | 保存置信度到 txt |
| `--save-crop` | `False` | 保存检测目标裁剪图像 |
| `--project` | `runs/detect` | 结果保存项目目录 |
| `--name` | `exp` | 结果保存实验名称 |

### 输出格式

检测结果保存在指定的项目目录下：
- **图像结果**: 标注了检测框的原图像
- **文本结果**: 每张图片对应的 `.txt` 文件，包含检测框坐标和置信度
- **裁剪图像**: `--save-crop` 选项保存检测到的目标区域

##  项目结构

```
yolov5_ghost/
├── data/                   # 数据集
│   ├── SafetyVests.v6/    # 安全背心数据集
│   │   ├── train/         # 训练集
│   │   ├── valid/         # 验证集
│   │   ├── test/          # 测试集
│   │   └── data.yaml      # 数据集配置
│   └── hyps/              # 超参数配置
├── models/                 # 模型配置文件
│   ├── yolov5s.yaml       # YOLOv5s 配置
│   ├── yolov5m.yaml       # YOLOv5m 配置
│   └── hub/               # 模型变体
├── runs/                   # 训练和检测结果
│   ├── train/             # 训练结果
│   │   └── exp3/          # 训练实验3
│   │       └── weights/   # 模型权重
│   │           ├── best.pt    # 最佳权重
│   │           └── last.pt    # 最后权重
│   ├── detect/            # 检测结果
│   └── val/               # 验证结果
├── utils/                  # 工具函数
├── paper/                  # 相关研究论文
├── train.py               # 训练脚本
├── detect.py              # 检测脚本
├── val.py                 # 验证脚本
├── export.py              # 模型导出脚本
├── requirements.txt       # pip 依赖包
├── environment.yml        # conda 环境配置
└── README.md              # 项目说明
```

## 📈 性能指标

### 模型性能
- **mAP@0.5**: xx.x%
- **mAP@0.5:0.95**: xx.x%
- **精确率 (Precision)**: xx.x%
- **召回率 (Recall)**: xx.x%
- **F1 分数**: xx.x

### 推理速度
- **GPU (RTX 3080)**: xx ms/image
- **CPU (Intel i7)**: xx ms/image
- **移动端**: xx ms/image

### 模型大小
- **YOLOv5s**: ~14MB
- **YOLOv5s Ghost**: ~10MB (轻量化版本)

## 🔧 配置文件

### 环境配置 (environment.yml)
```yaml
name: yolov5_ghost
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=1.8.0
  - torchvision>=0.9.0
  - numpy>=1.23.5
  - opencv>=4.1.1
  # ... 其他依赖
```

### 数据集配置 (data.yaml)
```yaml
path: ./  # dataset root dir
train: data/SafetyVests.v6/train/images
val: data/SafetyVests.v6/valid/images
test: data/SafetyVests.v6/test/images

nc: 2  # number of classes
names: ['NO-Safety Vest', 'Safety Vest']  # class names
```

### 模型配置 (yolov5s.yaml)
- 基于 YOLOv5s 架构
- 集成 Ghost 模块进行轻量化优化
- 针对安全背心检测任务优化

### 检测结果说明
使用训练好的模型检测后，结果保存在：
- **图像结果**: `runs/detect/test_results/` 目录
- **文本结果**: 每张图片对应的 `.txt` 文件，包含检测框坐标和置信度
- **裁剪图像**: `--save-crop` 选项保存检测到的目标区域

## 🎯 应用场景

- **建筑工地安全监控**: 实时检测工人是否佩戴安全背心
- **工厂安全管理**: 确保员工遵守安全规范
- **港口作业监控**: 检测码头工人安全装备佩戴情况
- **道路施工监控**: 监控路政工人安全防护
- **智能安防系统**: 集成到现有安防系统中

## 📚 相关研究

本项目基于以下研究成果：
- **论文**: 《基于深度学习的安全帽与反光衣检测研究》- 张学立
- **YOLOv5 官方仓库**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- **Ghost 模块**: 轻量化神经网络设计

## 📄 许可证

本项目采用 AGPL-3.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Ultralytics](https://ultralytics.com/) - YOLOv5 官方实现
- [Roboflow](https://roboflow.com/) - SafetyVests.v6 数据集提供
- 张学立 - 相关研究论文作者

---

⭐ 基于 YOLOv5 Ghost 的安全背心检测系统
