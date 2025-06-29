# YOLOv5 Ghost - 安全背心检测系统

![YOLOv5](https://img.shields.io/badge/YOLOv5-v7.0-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red)
![License](https://img.shields.io/badge/License-AGPL--3.0-lightgrey)
![GitHub release](https://img.shields.io/badge/release-v1.0.0-orange)
![GitHub stars](https://img.shields.io/github/stars/LJK666666666/yolov5_ghost?style=social)

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
- 🧪 **批量模型测试**: 支持多模型自动测试和性能对比
- 📈 **专业性能分析**: NO-Safety Vest召回率专门跟踪和分析
- 🎯 **智能结果管理**: 自动生成带时间戳的测试结果目录

## 🎯 YOLOv5s-Ghost 轻量化改进

本项目基于原始 YOLOv5 实现了多种轻量化改进方案，包括 GhostNet 模块、CA注意力机制和WIoU损失函数，旨在在保持检测精度的同时显著减少模型参数和计算量。

### 🚀 技术改进

#### 1. Ghost 模块原理
- **Ghost Convolution**: 使用少量卷积操作生成特征图，然后通过线性变换生成"Ghost"特征图
- **参数减少**: 相比标准卷积，Ghost 卷积可以减少约 50% 的参数和计算量
- **性能保持**: 在减少计算量的同时，保持接近原始模型的检测精度

#### 2. CA注意力机制（Coordinate Attention）
- **位置敏感**: 能够捕获跨通道信息和位置相关信息
- **轻量设计**: 计算开销小，适合移动端应用
- **特征增强**: 通过注意力权重增强重要特征，抑制无关信息

#### 3. WIoU损失函数（Wise IoU Loss）
- **动态聚焦**: 根据锚框质量动态调整损失权重
- **训练稳定**: 相比传统IoU损失，训练过程更加稳定
- **精度提升**: 特别适合小目标和遮挡目标的检测

#### 实现的模块
1. **GhostConv**: Ghost 卷积层，替代标准卷积
2. **GhostBottleneck**: Ghost 瓶颈模块，用于构建更深的网络
3. **C3Ghost**: 基于 Ghost Bottleneck 的 C3 模块
4. **CoordAtt**: 坐标注意力机制模块
5. **WIoU**: Wise IoU 损失函数实现

### 📊 模型配置对比

| 模型版本 | 架构特点 | 适用场景 | 配置文件 |
|----------|----------|----------|----------|
| **YOLOv5s** | 原始基线模型 | 高精度需求 | `models/yolov5s.yaml` |
| **YOLOv5s-Ghost_1** | 仅Ghost模块 | 基础轻量化 | `models/yolov5s-ghost_1.yaml` |
| **YOLOv5s-Ghost_2** | 仅CA注意力 | 精度优化 | `models/yolov5s-ghost_2.yaml` |
| **YOLOv5s-Ghost_12** | Ghost+CA组合 | 平衡性能 | `models/yolov5s-ghost_12.yaml` |
| **YOLOv5s-Ghost** | 最终推荐方案 | 生产部署 | `models/yolov5s-ghost.yaml` |

### 🔧 损失函数选择

| 损失函数 | 特点 | 使用场景 | 启用方式 |
|----------|------|----------|----------|
| **CIoU** | 传统IoU损失 | 一般目标检测 | 默认启用 |
| **WIoU** | 动态权重IoU | 小目标/遮挡检测 | `--box-loss wiou` |

### 🎛️ 超参数配置

| 配置文件 | 特点 | 适用模型 | 启用方式 |
|----------|------|----------|----------|
| **推荐超参数** | 优化配置 ⭐ **默认** | Ghost模型 | 默认使用（无需指定参数） |
| **自定义超参数** | 用户配置 | 所有模型 | `--hyp your_config.yaml` |

> 🎉 **新更新**: `hyp.recommand.yaml` 现已设为默认超参数配置，训练时无需手动指定即可享受优化效果！

### 🚀 性能优势
- **模型轻量化**: Ghost模块减少约 19% 的参数量
- **计算高效**: Ghost模块减少约 37% 的计算量
- **注意力增强**: CA注意力机制提升特征表达能力
- **损失优化**: WIoU损失函数提升小目标检测性能
- **部署友好**: 更小的模型体积，适合移动端和边缘设备
- **配置灵活**: 多种模型配置满足不同性能需求

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
- **许可证**: CC BY 4.0
- **数据来源**: [Roboflow Safety Vests Dataset v6](https://universe.roboflow.com/roboflow-universe-projects/safety-vests/dataset/6)

### 📥 数据集下载

**重要提示**: 由于数据集文件较大（约几百MB），未直接上传到GitHub仓库。请按以下步骤获取数据集：

#### 方法1：直接从 Roboflow 下载（推荐）
1. 访问数据集页面：[https://universe.roboflow.com/roboflow-universe-projects/safety-vests/dataset/6](https://universe.roboflow.com/roboflow-universe-projects/safety-vests/dataset/6)
2. 选择 **"YOLOv5 PyTorch"** 格式
3. 点击 **"Download zip to computer"**
4. 解压下载的文件到项目的 `data/SafetyVests.v6/` 目录

#### 方法2：使用 Roboflow Python SDK
```bash
# 安装 roboflow 库
pip install roboflow

# 下载数据集
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')  # 需要注册获取API Key
project = rf.workspace('roboflow-universe-projects').project('safety-vests')
dataset = project.version(6).download('yolov5', location='data/SafetyVests.v6')
"
```

#### 方法3：手动创建目录结构
如果暂时无法下载数据集，可以先创建目录结构用于测试：
```bash
mkdir -p data/SafetyVests.v6/{train,valid,test}/{images,labels}
```

### 数据集结构

下载并解压后，目录结构应该如下：

```
data/SafetyVests.v6/
├── train/
│   ├── images/     # 训练图像 (JPG格式)
│   └── labels/     # 训练标签 (TXT格式，YOLO标注)
├── valid/
│   ├── images/     # 验证图像
│   └── labels/     # 验证标签
├── test/
│   ├── images/     # 测试图像
│   └── labels/     # 测试标签
├── data.yaml       # 数据集配置文件
├── README.dataset.txt
└── README.roboflow.txt
```

### 📋 数据集信息

**检测类别**:
- `0`: NO-Safety Vest (未穿戴安全背心)
- `1`: Safety Vest (穿戴安全背心)

**数据集统计** (大约):
- 训练集: ~500+ 张图像
- 验证集: ~100+ 张图像  
- 测试集: ~100+ 张图像
- 标注格式: YOLO格式 (相对坐标)

## 🚀 快速开始

### 模型架构选择

本项目提供多种模型配置，根据不同需求选择：

```bash
# 1. 基线模型 - 原始 YOLOv5s
--cfg models/yolov5s.yaml

# 2. Ghost轻量化模型
--cfg models/yolov5s-ghost_1.yaml

# 3. CA注意力增强模型  
--cfg models/yolov5s-ghost_2.yaml

# 4. Ghost+CA组合模型
--cfg models/yolov5s-ghost_12.yaml

# 5. 最终推荐模型（等同于_12）
--cfg models/yolov5s-ghost.yaml
```

### 损失函数和超参数选择

```bash
# 使用WIoU损失函数（推荐小目标检测）
--box-loss wiou

# 🎉 好消息：推荐超参数配置现已默认启用！
# 以下命令会自动使用 data/hyps/hyp.recommand.yaml 配置
python train.py --data data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt

# 如需使用其他超参数配置，可手动指定
--hyp path/to/your/custom_hyp.yaml
```

> 💡 **重要更新**: 从 v1.1.0 开始，`hyp.recommand.yaml` 已设为默认超参数配置，无需手动指定即可享受优化的训练效果！

### 使用原始 YOLOv5s 模型

#### 训练原始 YOLOv5s

```bash
# 基础训练 - 原始 YOLOv5s
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_original

# 检测 - 使用原始模型
python detect.py --weights runs/train/yolov5s_original/weights/best.pt --source data/SafetyVests.v6/test/images --conf 0.25 --save-txt --project runs/detect --name yolov5s_results
```

### 使用 YOLOv5s-Ghost 轻量化模型系列

#### 1. 基础Ghost模型（仅Ghost模块）

```bash
# 训练Ghost_1模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_1.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_ghost_1

# 检测
python detect.py --weights runs/train/yolov5s_ghost_1/weights/best.pt --source data/SafetyVests.v6/test/images --conf 0.25 --save-txt --project runs/detect --name yolov5s_ghost_1_results
```

#### 2. CA注意力模型（仅注意力机制）

```bash
# 训练Ghost_2模型（CA注意力）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_2.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_ghost_2

# 检测
python detect.py --weights runs/train/yolov5s_ghost_2/weights/best.pt --source data/SafetyVests.v6/test/images --conf 0.25 --save-txt --project runs/detect --name yolov5s_ghost_2_results
```

#### 3. 组合模型（Ghost + CA，推荐）

```bash
# 训练Ghost_12组合模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_ghost_12

# 使用WIoU损失函数训练
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --box-loss wiou --project runs/train --name yolov5s_ghost_12_wiou

# 使用推荐超参数训练
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --hyp data/hyps/hyp.recommand.yaml --project runs/train --name yolov5s_ghost_12_hyp

# 组合使用所有优化（推荐配置）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --box-loss wiou --hyp data/hyps/hyp.recommand.yaml --project runs/train --name yolov5s_ghost_12_full

# 检测
python detect.py --weights runs/train/yolov5s_ghost_12/weights/best.pt --source data/SafetyVests.v6/test/images --conf 0.25 --save-txt --project runs/detect --name yolov5s_ghost_12_results
```

#### 4. 最终推荐模型

```bash
# 使用最终推荐模型配置（等同于ghost_12）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --box-loss wiou --hyp data/hyps/hyp.recommand.yaml --project runs/train --name yolov5s_ghost_final

# 检测
python detect.py --weights runs/train/yolov5s_ghost_final/weights/best.pt --source data/SafetyVests.v6/test/images --conf 0.25 --save-txt --project runs/detect --name yolov5s_ghost_final_results
```

### 模型对比实验

#### 完整对比实验（推荐）

```bash
# 1. 基线模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 100 --name baseline_comparison

# 2. Ghost轻量化模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_1.yaml --weights yolov5s.pt --epochs 100 --name ghost_1_comparison

# 3. CA注意力模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_2.yaml --weights yolov5s.pt --epochs 100 --name ghost_2_comparison

# 4. Ghost+CA组合模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --epochs 100 --name ghost_12_comparison

# 5. Ghost+CA+WIoU组合（完整优化）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --epochs 100 --box-loss wiou --name ghost_12_wiou_comparison

# 6. 最终推荐配置
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt --epochs 100 --box-loss wiou --hyp data/hyps/hyp.recommand.yaml --name final_optimized_comparison
```

#### 快速对比实验

```bash
# 基线 vs 最优配置对比
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 100 --name baseline
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt --epochs 100 --box-loss wiou --hyp data/hyps/hyp.recommand.yaml --name optimized
```

> 💡 **Linux/macOS 并行训练**: 在命令末尾加 `&` 可以并行运行

### 使用训练好的模型检测测试集

使用项目中已训练好的权重：

```bash
# 使用训练好的最佳权重检测测试集
python detect.py --weights runs/train/exp3/weights/best.pt --source data/SafetyVests.v6/test/images --conf 0.25 --save-txt --save-conf --project runs/test --name safety_vest_test

# 检测单张测试图片
python detect.py --weights runs/train/exp3/weights/best.pt --source data/SafetyVests.v6/test/images/image_name.jpg --conf 0.25 --save-txt

# 批量检测并保存详细结果
python detect.py --weights runs/train/exp3/weights/best.pt --source data/SafetyVests.v6/test/images --conf 0.25 --iou-thres 0.45 --save-txt --save-conf --save-crop --line-thickness 2 --project runs/detect --name test_results
```

### 模型验证和评估

```bash
# 在验证集上评估模型性能
python val.py --weights runs/train/exp3/weights/best.pt --data data/SafetyVests.v6/data.yaml --img 640 --conf 0.001 --iou 0.6 --project runs/val --name exp

# 在测试集上评估（如果测试集有标签）
python val.py --weights runs/train/exp3/weights/best.pt --data data/SafetyVests.v6/data.yaml --task test --img 640
```

## 🎓 训练模型

### 模型配置说明

根据`models/model.md`说明，项目提供以下模型配置：

- **yolov5s.yaml**: 基线方案（原始YOLOv5s）
- **yolov5s-ghost_1.yaml**: 添加Ghost模块（GhostConv和C3Ghost）
- **yolov5s-ghost_2.yaml**: 添加CA注意力机制
- **yolov5s-ghost_12.yaml**: 同时包含Ghost模块和CA注意力机制
- **yolov5s-ghost.yaml**: 最终方案（目前和yolov5s-ghost_12.yaml相同）

### 在 SafetyVests.v6 数据集上训练

#### 1. 基线模型训练

```bash
# 基础训练命令 - 原始模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_baseline
```

#### 2. Ghost轻量化模型训练

```bash
# Ghost模块训练（_1版本）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_1.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_ghost_1
```

#### 3. CA注意力模型训练

```bash
# CA注意力训练（_2版本）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_2.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_ghost_2
```

#### 4. Ghost+CA组合模型训练

```bash
# 基础组合训练（_12版本）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --project runs/train --name yolov5s_ghost_12

# 使用WIoU损失函数
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --box-loss wiou --project runs/train --name yolov5s_ghost_12_wiou

# 使用推荐超参数
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --hyp data/hyps/hyp.recommand.yaml --project runs/train --name yolov5s_ghost_12_hyp
```

#### 5. 最终推荐配置训练

**完整优化训练 (多行格式便于理解):**
```bash
# 最终推荐配置 - 包含所有优化
python train.py \
    --data data/SafetyVests.v6/data.yaml \
    --cfg models/yolov5s-ghost.yaml \
    --weights yolov5s.pt \
    --batch-size 16 \
    --epochs 100 \
    --img-size 640 \
    --device 0 \
    --box-loss wiou \
    --hyp data/hyps/hyp.recommand.yaml \
    --project runs/train \
    --name yolov5s_ghost_final
```

**PowerShell格式:**
```powershell
# Windows PowerShell 用户
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --img-size 640 --device 0 --box-loss wiou --hyp data/hyps/hyp.recommand.yaml --project runs/train --name yolov5s_ghost_final
```

### 训练参数说明

#### 新增参数

| 参数 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| `--box-loss` | `ciou`, `wiou` | `ciou` | 边界框损失函数类型 |
| `--hyp` | 超参数文件路径 | 内置默认值 | 自定义超参数配置文件 |

#### 推荐参数组合

**高精度训练:**
```bash
--cfg models/yolov5s.yaml --epochs 200 --batch-size 32
```

**轻量化训练:**
```bash
--cfg models/yolov5s-ghost_1.yaml --epochs 150 --batch-size 16
```

**平衡性能训练（推荐）:**
```bash
--cfg models/yolov5s-ghost_12.yaml --box-loss wiou --hyp data/hyps/hyp.recommand.yaml --epochs 100 --batch-size 16
```

### 恢复训练

```bash
# 恢复原始模型训练
python train.py --resume runs/train/yolov5s_baseline/weights/last.pt

# 恢复Ghost组合模型训练
python train.py --resume runs/train/yolov5s_ghost_12/weights/last.pt

# 恢复最终推荐模型训练
python train.py --resume runs/train/yolov5s_ghost_final/weights/last.pt
```

### 训练监控

```bash
# 启动 TensorBoard 查看训练进度
tensorboard --logdir runs/train --port 6006
```

### 模型验证

```bash
# 验证原始模型
python val.py --weights runs/train/yolov5s_baseline/weights/best.pt --data data/SafetyVests.v6/data.yaml --img 640 --conf 0.001 --iou 0.6 --project runs/val --name yolov5s_val

# 验证Ghost_1模型
python val.py --weights runs/train/yolov5s_ghost_1/weights/best.pt --data data/SafetyVests.v6/data.yaml --img 640 --conf 0.001 --iou 0.6 --project runs/val --name yolov5s_ghost_1_val

# 验证Ghost_2模型
python val.py --weights runs/train/yolov5s_ghost_2/weights/best.pt --data data/SafetyVests.v6/data.yaml --img 640 --conf 0.001 --iou 0.6 --project runs/val --name yolov5s_ghost_2_val

# 验证Ghost_12模型
python val.py --weights runs/train/yolov5s_ghost_12/weights/best.pt --data data/SafetyVests.v6/data.yaml --img 640 --conf 0.001 --iou 0.6 --project runs/val --name yolov5s_ghost_12_val

# 验证最终推荐模型
python val.py --weights runs/train/yolov5s_ghost_final/weights/best.pt --data data/SafetyVests.v6/data.yaml --img 640 --conf 0.001 --iou 0.6 --project runs/val --name yolov5s_ghost_final_val
```

训练完成后，模型权重将保存在对应的 `runs/train/实验名称/weights/` 目录下

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

### 训练参数（新增）

| 参数 | 选项 | 默认值 | 描述 |
|------|------|--------|------|
| `--cfg` | 模型配置文件 | - | 选择不同的模型架构 |
| `--box-loss` | `ciou`, `wiou` | `ciou` | 边界框损失函数类型 |
| `--hyp` | 超参数文件路径 | 内置默认 | 自定义超参数配置 |

### 模型配置选择

| 配置文件 | 架构特点 | 适用场景 |
|----------|----------|----------|
| `models/yolov5s.yaml` | 原始基线 | 高精度需求 |
| `models/yolov5s-ghost_1.yaml` | Ghost轻量化 | 基础优化 |
| `models/yolov5s-ghost_2.yaml` | CA注意力 | 精度增强 |
| `models/yolov5s-ghost_12.yaml` | Ghost+CA | 平衡性能 |
| `models/yolov5s-ghost.yaml` | 最终方案 | 生产推荐 |

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
│       ├── hyp.recommand.yaml  # 推荐超参数配置 ✨
│       └── hyp.*.yaml     # 其他超参数配置
├── models/                 # 模型配置文件
│   ├── yolov5s.yaml       # YOLOv5s 原始配置（基线）
│   ├── yolov5s-ghost_1.yaml   # Ghost模块版本 ✨
│   ├── yolov5s-ghost_2.yaml   # CA注意力版本 ✨
│   ├── yolov5s-ghost_12.yaml  # Ghost+CA组合版本 ✨
│   ├── yolov5s-ghost.yaml # 最终推荐版本 ✨
│   ├── yolov5m.yaml       # YOLOv5m 配置
│   ├── common.py          # 包含所有模块实现 ✨
│   │   ├── GhostConv      # Ghost 卷积层
│   │   ├── GhostBottleneck# Ghost 瓶颈模块
│   │   ├── C3Ghost        # Ghost C3 模块
│   │   └── CoordAtt       # CA坐标注意力模块
│   ├── model.md           # 模型配置说明文档 ✨
│   └── hub/               # 模型变体
├── runs/                   # 训练和检测结果
│   ├── train/             # 训练结果
│   │   ├── yolov5s_baseline/          # 原始模型训练
│   │   ├── yolov5s_ghost_1/           # Ghost模块训练
│   │   ├── yolov5s_ghost_2/           # CA注意力训练
│   │   ├── yolov5s_ghost_12/          # Ghost+CA训练
│   │   ├── yolov5s_ghost_12_wiou/     # 使用WIoU损失训练
│   │   ├── yolov5s_ghost_final/       # 最终推荐配置训练
│   │   └── weights/       # 模型权重
│   │       ├── best.pt    # 最佳权重
│   │       └── last.pt    # 最后权重
│   ├── detect/            # 检测结果
│   └── val/               # 验证结果
├── utils/                  # 工具函数
│   ├── loss.py            # 损失函数（包含WIoU实现）✨
│   └── ...                # 其他工具函数
├── paper/                  # 相关研究论文
│   └── ghost.pdf          # GhostNet 论文 📚
├── train.py               # 训练脚本（支持--box-loss和--hyp参数）✨
├── detect.py              # 检测脚本
├── val.py                 # 验证脚本
├── export.py              # 模型导出脚本
├── test_all_models.py     # 批量模型测试脚本 ✨
├── test_models_usage.md   # 模型测试使用说明 ✨
├── ENHANCEMENT_SUMMARY.md # 功能增强总结 ✨
├── NO_SAFETY_VEST_RECALL_ENHANCEMENT.md # NO-Safety Vest召回率增强说明 ✨
├── requirements.txt       # pip 依赖包
├── environment.yml        # conda 环境配置
└── README.md              # 项目说明
```

### 🔧 核心修改文件

#### 1. 模型配置文件（新增多个版本）
- **models/yolov5s-ghost_1.yaml**: 仅包含Ghost模块的轻量化版本
- **models/yolov5s-ghost_2.yaml**: 仅包含CA注意力机制的版本
- **models/yolov5s-ghost_12.yaml**: 同时包含Ghost模块和CA注意力机制
- **models/yolov5s-ghost.yaml**: 最终推荐方案（等同于_12版本）

#### 2. models/common.py (扩展)
包含完整的轻量化模块实现：
- `GhostConv`: 实现 Ghost 卷积操作
- `GhostBottleneck`: Ghost 瓶颈结构
- `C3Ghost`: 基于 Ghost Bottleneck 的 C3 模块
- `CoordAtt`: 坐标注意力机制模块

#### 3. utils/loss.py (修改)
- 添加了完整的 `WIoU` 损失函数实现
- 支持在训练时通过 `--box-loss wiou` 参数启用

#### 4. train.py (修改)
- 新增 `--box-loss` 参数支持CIoU和WIoU损失函数选择
- 完善超参数文件支持

#### 5. data/hyps/hyp.recommand.yaml (新增)
- 针对Ghost模型优化的推荐超参数配置
- 包含数据增强、学习率等优化参数

#### 3. 主干网络架构对比

| 层级 | 原始 YOLOv5s | Ghost_1 | Ghost_2 | Ghost_12/Final | 说明 |
|------|---------------|---------|---------|----------------|------|
| P1/2 | Conv | GhostConv | Conv | GhostConv | 第一层卷积 |
| P2/4 | Conv | GhostConv | Conv | GhostConv | 第二层卷积 |
| CSP1 | C3 | C3Ghost | C3 | C3Ghost | 第一个 CSP 模块 |
| CA1 | - | - | CoordAtt | CoordAtt | 第一个注意力层 |
| P3/8 | Conv | GhostConv | Conv | GhostConv | 第三层卷积 |
| CSP2 | C3 | C3Ghost | C3 | C3Ghost | 第二个 CSP 模块 |
| CA2 | - | - | CoordAtt | CoordAtt | 第二个注意力层 |
| P4/16 | Conv | GhostConv | Conv | GhostConv | 第四层卷积 |
| CSP3 | C3 | C3Ghost | C3 | C3Ghost | 第三个 CSP 模块 |
| P5/32 | Conv | GhostConv | Conv | GhostConv | 第五层卷积 |
| CSP4 | C3 | C3Ghost | C3 | C3Ghost | 第四个 CSP 模块 |
| SPPF | SPPF | SPPF | SPPF | SPPF | 空间金字塔池化（保持不变） |
| CA3 | - | - | CoordAtt | CoordAtt | 第三个注意力层 |
| **Head CA** | - | - | CoordAtt×3 | CoordAtt×3 | 检测头注意力 |

#### 4. 损失函数对比

| 损失函数 | 特点 | 数学原理 | 适用场景 |
|----------|------|----------|----------|
| **CIoU** | 考虑距离、重叠、比例 | Complete IoU | 一般目标检测 |
| **WIoU** | 动态权重聚焦 | Wise IoU with focus | 小目标、遮挡检测 |

### 超参数调优

#### 统一超参数配置（便于对比）

> 🎯 **重要**: 所有模型均采用默认的推荐超参数配置，确保公平对比

**所有模型统一配置:**
- 学习率: 0.01 (推荐配置默认值)
- 训练轮次: 100 (标准对比轮次)
- 批次大小: 16 (统一批次大小)
- 数据增强: 推荐配置 (默认启用)
- 超参数文件: `hyp.recommand.yaml` (自动使用)

**模型差异仅在于架构:**
- **YOLOv5s**: 原始基线架构
- **Ghost_1**: 仅Ghost轻量化模块
- **Ghost_2**: 仅CA注意力机制
- **Ghost_12**: Ghost + CA组合优化

这样可以纯粹对比不同架构的性能差异，排除超参数影响。

#### WIoU损失函数使用建议
```bash
# 适合使用WIoU的场景
--box-loss wiou  # 当遇到以下情况时推荐使用：
# 1. 小目标较多的场景
# 2. 目标密集重叠的场景  
# 3. 背景复杂的场景
# 4. 训练收敛困难时
```

#### 推荐超参数文件说明
`data/hyps/hyp.recommand.yaml` 包含：
- 优化的学习率调度
- 增强的数据增强参数
- 适配Ghost模型的权重衰减
- 安全背心检测任务的特定优化

### 部署优化

#### 模型导出

```bash
# 导出 ONNX 格式 - 原始模型
python export.py --weights runs/train/yolov5s_baseline/weights/best.pt --include onnx --img-size 640

# 导出 ONNX 格式 - Ghost_1轻量化模型
python export.py --weights runs/train/yolov5s_ghost_1/weights/best.pt --include onnx --img-size 640

# 导出 ONNX 格式 - Ghost_2注意力模型
python export.py --weights runs/train/yolov5s_ghost_2/weights/best.pt --include onnx --img-size 640

# 导出 ONNX 格式 - Ghost_12组合模型（推荐）
python export.py --weights runs/train/yolov5s_ghost_12/weights/best.pt --include onnx --img-size 640

# 导出 ONNX 格式 - 最终推荐模型
python export.py --weights runs/train/yolov5s_ghost_final/weights/best.pt --include onnx --img-size 640
```

#### 移动端部署

```bash
# 导出 TensorRT - 适合 NVIDIA 设备（Ghost轻量化模型推荐）
python export.py --weights runs/train/yolov5s_ghost_1/weights/best.pt --include engine --device 0

# 导出 CoreML - 适合 iOS 设备（Ghost轻量化模型推荐）
python export.py --weights runs/train/yolov5s_ghost_1/weights/best.pt --include coreml

# 导出 TensorRT - 平衡性能模型
python export.py --weights runs/train/yolov5s_ghost_12/weights/best.pt --include engine --device 0

# 导出 CoreML - 平衡性能模型
python export.py --weights runs/train/yolov5s_ghost_12/weights/best.pt --include coreml
```

### 性能调优技巧

#### 推理优化
1. **批处理**: 对于批量图像处理，使用更大的 batch size
2. **输入尺寸**: 根据精度需求调整输入图像尺寸（416, 512, 640）
3. **后处理**: 调整 NMS 阈值平衡速度和精度
4. **模型选择**: 
   - 速度优先: 选择 Ghost_1
   - 精度优先: 选择 Ghost_2  
   - 平衡性能: 选择 Ghost_12

#### 内存优化
1. **半精度推理**: 使用 FP16 减少内存占用
2. **模型剪枝**: 进一步减少模型大小
3. **量化**: 使用 INT8 量化提升推理速度
4. **架构选择**: Ghost系列模型天然内存友好

#### 训练优化
1. **损失函数**: 小目标多时使用 WIoU 损失
2. **数据增强**: 使用推荐超参数配置  
3. **学习率**: 根据模型架构调整学习率策略
4. **训练轮次**: Ghost模型建议更多训练轮次

## 📈 性能指标

### 模型架构对比

| 指标 | YOLOv5s | Ghost_1 | Ghost_2 | Ghost_12 | 说明 |
|------|---------|---------|---------|----------|------|
| **Ghost模块** | ❌ | ✅ | ❌ | ✅ | 轻量化卷积 |
| **CA注意力** | ❌ | ❌ | ✅ | ✅ | 坐标注意力 |
| **预期参数量** | 7.2M | ~5.8M | ~7.5M | ~6.0M | 理论估算 |
| **预期计算量** | 16.5G | ~10.3G | ~17.2G | ~11.0G | 理论估算 |
| **适用场景** | 基线 | 轻量化 | 精度优先 | 平衡性能 | 部署建议 |

### 损失函数对比

| 损失函数 | 特点 | 优势 | 适用场景 |
|----------|------|------|----------|
| **CIoU** | 完整IoU | 训练稳定 | 一般目标检测 |
| **WIoU** | 智能权重 | 聚焦难样本 | 小目标/密集检测 |

### 超参数配置对比

| 配置 | 特点 | 数据增强强度 | 适用模型 |
|------|------|--------------|----------|
| **默认超参数** | 通用配置 | 标准 | 所有模型 |
| **推荐超参数** | 优化配置 | 增强 | Ghost系列 |

*注：具体性能数据将在训练完成后更新*

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

### 模型配置对比

#### 原始 YOLOv5s (models/yolov5s.yaml)
```yaml
# YOLOv5 v6.0 backbone
backbone:
  [
    [-1, 1, Conv, [64, 6, 2, 2]],        # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]],          # 1-P2/4
    [-1, 3, C3, [128]],                  # 2
    [-1, 1, Conv, [256, 3, 2]],          # 3-P3/8
    [-1, 6, C3, [256]],                  # 4
    [-1, 1, Conv, [512, 3, 2]],          # 5-P4/16
    [-1, 9, C3, [512]],                  # 6
    [-1, 1, Conv, [1024, 3, 2]],         # 7-P5/32
    [-1, 3, C3, [1024]],                 # 8
    [-1, 1, SPPF, [1024, 5]],            # 9
  ]
```

#### YOLOv5s-Ghost 模型变体对比 (models/yolov5s-ghost*.yaml) ✨

**Ghost_1 版本:**
```yaml
# 仅包含Ghost模块的轻量化版本
backbone:
  [
    [-1, 1, GhostConv, [64, 6, 2, 2]],   # Ghost卷积替换
    [-1, 1, GhostConv, [128, 3, 2]],     # Ghost卷积替换
    [-1, 3, C3Ghost, [128]],             # Ghost C3替换
    # ... 其他层使用Ghost模块
  ]
```

**Ghost_2 版本:**
```yaml
# 仅包含CA注意力机制的版本
backbone:
  [
    [-1, 1, Conv, [64, 6, 2, 2]],        # 保持标准卷积
    [-1, 1, Conv, [128, 3, 2]],          # 保持标准卷积
    [-1, 3, C3, [128]],                  # 保持标准C3
    [-1, 1, CoordAtt, [256]],            # 添加CA注意力 ✨
    # ... 在关键位置添加注意力机制
  ]
```

**Ghost_12 版本（推荐）:**
```yaml
# 同时包含Ghost模块和CA注意力机制
backbone:
  [
    [-1, 1, GhostConv, [64, 6, 2, 2]],   # Ghost卷积
    [-1, 1, GhostConv, [128, 3, 2]],     # Ghost卷积
    [-1, 3, C3Ghost, [128]],             # Ghost C3
    [-1, 1, CoordAtt, [256]],            # CA注意力 ✨
    # ... 结合两种优化技术
  ]
```

### WIoU损失函数实现 (utils/loss.py) ✨

```python
class WIoU:
    """
    Wise-IoU loss function implementation
    Paper: https://arxiv.org/abs/2301.10051
    """
    def __init__(self, pred, target, eps=1e-7, alpha=2.0, beta=4.0):
        self.eps = eps
        self.alpha = alpha  
        self.beta = beta
        self.pred = pred
        self.target = target
        self.iou = bbox_iou(pred, target, xywh=True, CIoU=False).squeeze()

    @property
    def wiou(self):
        """Calculate WIoU loss with dynamic focusing mechanism"""
        # 计算中心点距离
        dist = torch.sum((self.pred[:, :2] - self.target[:, :2]) ** 2, dim=1)
        
        # 计算包围框尺寸
        # ... 详细实现见源码
        
        # R_WIoU 计算
        r_wiou = torch.exp(dist / (cw ** 2 + ch ** 2 + self.eps))
        
        # 最终WIoU损失计算
        beta = (self.iou.detach() / self.alpha).pow(self.beta)
        loss_wiou = r_wiou * (1 - self.iou) * beta
        return loss_wiou.mean()
```

### CoordAtt注意力机制实现 (models/common.py) ✨

```python
class CoordAtt(nn.Module):
    """
    Coordinate Attention mechanism
    Paper: Coordinate Attention for Efficient Mobile Network Design
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # X和Y方向的池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        # 分别处理H和W方向的注意力
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 分别在H和W方向进行池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # 连接和处理
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分离H和W方向的注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # 应用注意力
        out = identity * a_w * a_h
        return out
```

### 推荐超参数配置 (data/hyps/hyp.recommand.yaml) ✨

```yaml
# 针对Ghost模型优化的超参数配置
lr0: 0.01  # 初始学习率
lrf: 0.1   # 最终学习率比例
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# 损失函数权重
box: 0.05  # 边界框损失权重
cls: 0.5   # 分类损失权重
obj: 1.0   # 目标检测损失权重

# 数据增强优化（针对安全背心检测）
hsv_h: 0.015    # 色调增强
hsv_s: 0.7      # 饱和度增强  
hsv_v: 0.4      # 亮度增强
degrees: 10.0   # 旋转角度（增加）
translate: 0.1  # 平移
scale: 0.5      # 缩放
shear: 2.0      # 剪切变换（增加）
perspective: 0.0 # 透视变换
```

### 使用说明

#### 模型选择参数
- `--cfg models/yolov5s.yaml`: 使用原始 YOLOv5s（基线）
- `--cfg models/yolov5s-ghost_1.yaml`: 使用Ghost轻量化版本
- `--cfg models/yolov5s-ghost_2.yaml`: 使用CA注意力版本  
- `--cfg models/yolov5s-ghost_12.yaml`: 使用Ghost+CA组合版本
- `--cfg models/yolov5s-ghost.yaml`: 使用最终推荐版本

#### 损失函数选择参数
- `--box-loss ciou`: 使用CIoU损失函数（默认）
- `--box-loss wiou`: 使用WIoU损失函数（推荐小目标检测）

#### 超参数配置参数
- `--hyp data/hyps/hyp.recommand.yaml`: 使用推荐超参数（Ghost模型优化）

#### 完整命令示例
```bash
# 最优配置训练
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt --box-loss wiou --hyp data/hyps/hyp.recommand.yaml

# 基线对比训练
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt

# 轻量化训练
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_1.yaml --weights yolov5s.pt

# 注意力增强训练
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_2.yaml --weights yolov5s.pt
```

#### 快速开始 (单行命令)
```bash
# 训练基线模型
python train.py --cfg models/yolov5s.yaml --data data/SafetyVests.v6/data.yaml --weights yolov5s.pt

# 训练Ghost轻量化模型
python train.py --cfg models/yolov5s-ghost_1.yaml --data data/SafetyVests.v6/data.yaml --weights yolov5s.pt

# 训练CA注意力模型
python train.py --cfg models/yolov5s-ghost_2.yaml --data data/SafetyVests.v6/data.yaml --weights yolov5s.pt

# 训练Ghost+CA组合模型
python train.py --cfg models/yolov5s-ghost_12.yaml --data data/SafetyVests.v6/data.yaml --weights yolov5s.pt

# 训练最终推荐模型（完整优化）
python train.py --cfg models/yolov5s-ghost.yaml --data data/SafetyVests.v6/data.yaml --weights yolov5s.pt --box-loss wiou --hyp data/hyps/hyp.recommand.yaml
```

#### 完整工作流程
```bash
# 1. 训练基线模型（对比用）
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 100 --name baseline

# 2. 训练Ghost轻量化模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_1.yaml --weights yolov5s.pt --epochs 120 --name ghost_1

# 3. 训练CA注意力模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_2.yaml --weights yolov5s.pt --epochs 110 --name ghost_2

# 4. 训练Ghost+CA组合模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost_12.yaml --weights yolov5s.pt --epochs 100 --name ghost_12

# 5. 训练最终优化模型
python train.py --data data/SafetyVests.v6/data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt --epochs 100 --box-loss wiou --hyp data/hyps/hyp.recommand.yaml --name final

# 6. 验证所有模型
python val.py --weights runs/train/baseline/weights/best.pt --data data/SafetyVests.v6/data.yaml
python val.py --weights runs/train/ghost_1/weights/best.pt --data data/SafetyVests.v6/data.yaml
python val.py --weights runs/train/ghost_2/weights/best.pt --data data/SafetyVests.v6/data.yaml
python val.py --weights runs/train/ghost_12/weights/best.pt --data data/SafetyVests.v6/data.yaml
python val.py --weights runs/train/final/weights/best.pt --data data/SafetyVests.v6/data.yaml

# 7. 检测测试（使用最佳模型）
python detect.py --weights runs/train/final/weights/best.pt --source data/SafetyVests.v6/test/images --save-txt --save-conf

# 8. 导出部署模型
python export.py --weights runs/train/final/weights/best.pt --include onnx --img-size 640
```

> 💡 **提示**: 所有命令都采用跨平台兼容格式，可在Windows、Linux、macOS上直接运行

### 未来改进方向
🚀 **进一步轻量化**: 结合知识蒸馏和模型剪枝技术  
🚀 **自动化调优**: 自动搜索最优的Ghost和CA模块配置  
🚀 **多任务扩展**: 扩展到其他YOLO任务（分割、分类、姿态估计）  
🚀 **硬件优化**: 针对特定硬件平台的专门优化（ARM、NPU等）  
🚀 **新技术融合**: 集成更多SOTA轻量化技术（MobileNet、EfficientNet等）  
🚀 **端到端优化**: 从数据预处理到后处理的全链路优化  

## 🎬 视频流检测

您可以使用 `tools/video.py` 脚本进行实时视频流检测。

### 摄像头实时检测

```bash
python tools/video.py --weights models_trained/Ghost_e10_0626/weights/best.pt --source 0
```

- `--weights`: 指定训练好的模型权重路径。
- `--source 0`: 使用默认摄像头。

### 视频文件检测

```bash
python tools/video.py --weights models_trained/Ghost_e10_0626/weights/best.pt --source ./data/videos/your_video.mp4
```

- `--source`: 指定视频文件路径。

在检测窗口按 `q` 键退出。

---

⭐ **基于 YOLOv5 的多模块轻量化安全背心检测系统**  
🎯 **高效 · 轻量 · 精准 · 易用 · 可配置**  
🔧 **Ghost轻量化 + CA注意力 + WIoU损失 + 优化超参数**

## 🧪 模型测试与评估

### 批量模型测试工具

本项目提供了强大的批量模型测试工具 `test_all_models.py`，支持自动测试多个模型并生成详细的性能对比报告。

#### 🎯 主要功能
- ✅ **多模型批量测试**: 自动测试指定文件夹下的所有模型
- ✅ **灵活的模型选择**: 支持选择 best.pt 或 last.pt 模型
- ✅ **训练文件夹选择**: 支持测试不同训练轮数的模型（train200epoch, train300epoch等）
- ✅ **NO-Safety Vest召回率专门跟踪**: 重点关注安全背心检测的关键指标
- ✅ **自定义检测阈值**: 支持自定义置信度和IoU阈值
- ✅ **智能输出目录**: 自动生成带时间戳和训练信息的结果目录
- ✅ **详细性能报告**: 生成汇总报告和JSON格式的详细结果

#### 🚀 快速开始

```bash
# 列出可用的训练文件夹
python test_all_models.py --list-folders

# 测试train200epoch文件夹下的所有best模型（默认）
python test_all_models.py

# 测试train300epoch文件夹下的所有last模型
python test_all_models.py --train-folder train300epoch --model-type last

# 使用自定义阈值测试
python test_all_models.py --train-folder train300epoch --model-type best --conf-thres 0.25 --iou-thres 0.7
```

#### 📊 输出结果

测试完成后会生成以下结果：

```
runs/train300epoch_test_best_20250629_161302/
├── summary_report.txt          # 汇总报告
├── detailed_results.json       # 详细JSON结果
├── error_images/               # 预测错误的图片
│   ├── model1/
│   └── model2/
├── model1/                     # 各模型的详细结果
│   ├── results.csv
│   ├── labels/
│   └── ...
└── model2/
    └── ...
```

#### 📈 性能对比示例

汇总报告包含两个重要的性能对比表格：

**整体模型性能对比:**
```
模型名称              Precision    Recall       mAP@0.5      mAP@0.5:0.95
yolov5s-ghost_1_     0.8390       0.8400       0.8670       0.5350
yolov5s-ghost_2_     0.8930       0.8420       0.8930       0.5810
yolov5s_            0.8890       0.8300       0.8890       0.5720
```

**NO-Safety Vest 类别性能对比:**
```
模型名称              Precision    Recall       mAP@0.5      mAP@0.5:0.95
yolov5s-ghost_1_     0.8090       0.8090       0.8310       0.4420
yolov5s-ghost_2_     0.8420       0.8390       0.8560       0.4890
yolov5s_            0.8390       0.8390       0.8560       0.4890
```

#### 🔧 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-type` | str | `best` | 选择模型类型：`best` 或 `last` |
| `--train-folder` | str | `train200epoch` | 训练文件夹名称 |
| `--conf-thres` | float | `0.001` | 置信度阈值（0.0-1.0） |
| `--iou-thres` | float | `0.6` | IoU阈值（0.0-1.0） |
| `--list-folders` | flag | - | 列出所有可用的训练文件夹 |

#### 💡 使用建议

1. **验证模式**: 使用低置信度阈值（0.001）评估完整召回率
2. **实际应用**: 使用较高置信度阈值（0.25-0.5）获得可靠检测
3. **模型对比**: 同时测试不同训练轮数和模型类型，找出最佳配置
4. **关注NO-Safety Vest召回率**: 这是安全检测应用的关键指标

详细使用说明请参考：[test_models_usage.md](test_models_usage.md)

## 📝 更新日志

### v1.2.0 (2025-06-29)

#### 🧪 模型测试与评估功能
- **批量模型测试工具**: 新增 `test_all_models.py` 脚本，支持批量测试多个模型
- **训练文件夹选择**: 支持选择不同训练文件夹的模型进行测试
- **NO-Safety Vest召回率专门跟踪**: 重点关注安全背心检测的关键指标
- **智能输出目录命名**: 自动生成 `{train_folder}_test_{model_type}_{timestamp}` 格式的结果目录
- **详细性能报告**: 生成包含整体性能和NO-Safety Vest专门性能的汇总报告
- **自动文件夹检测**: 自动扫描和验证可用的训练文件夹

#### 🔧 功能增强
- **命令行参数支持**: 支持通过命令行参数控制模型类型、训练文件夹、置信度阈值等
- **错误处理优化**: 增强的错误处理和用户友好的提示信息
- **解析功能修复**: 修复NO-Safety Vest召回率解析问题，确保准确的性能指标

#### 📊 测试结果示例
```bash
# 不同训练文件夹模型对比结果
train200epoch + best + conf=0.001: 最佳mAP@0.5=0.9050, NO-Safety Vest召回率=0.8280
train300epoch + best + conf=0.25:  最佳mAP@0.5=0.8930, NO-Safety Vest召回率=0.8390
```

### v1.1.0 (2025-06-27)

#### 🔧 超参数优化
- **默认超参数配置更新**: 将 `hyp.recommand.yaml` 设置为默认的数据增强配置
  - 主训练脚本 `train.py` 现在默认使用优化的超参数配置
  - 分割训练脚本 `segment/train.py` 同步更新默认配置
  - 无需手动指定 `--hyp` 参数即可享受优化的训练效果

#### 📈 性能改进
- **增强的数据增强**: 包含更优的旋转、剪切、混合等增强参数
- **训练稳定性**: 针对 Ghost 模型优化的学习率和权重衰减配置
- **小目标检测**: 改进的 HSV 和几何变换参数，提升小目标检测性能

#### 🚀 使用便利性
- **简化命令**: 训练命令更加简洁，默认配置即为最优
- **向后兼容**: 仍支持通过 `--hyp` 参数指定自定义超参数文件
- **一致性**: 主训练和分割训练使用相同的优化配置

#### 示例更新
```bash
# 之前需要手动指定
python train.py --data data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt --hyp data/hyps/hyp.recommand.yaml

# 现在可以直接使用（自动使用优化配置）
python train.py --data data.yaml --cfg models/yolov5s-ghost.yaml --weights yolov5s.pt
```

### v1.0.0 (2025-06-25)

#### 🎯 初始发布
- **Ghost 轻量化模块**: 基于 GhostNet 的模型轻量化
- **CA 注意力机制**: 坐标注意力增强特征表达
- **WIoU 损失函数**: 智能 IoU 损失优化训练
- **多模型配置**: 提供 5 种不同的模型配置选择
- **完整文档**: 详细的安装、训练、检测指南

---

⭐ **基于 YOLOv5 的多模块轻量化安全背心检测系统**  
🎯 **高效 · 轻量 · 精准 · 易用 · 可配置**  
🔧 **Ghost轻量化 + CA注意力 + WIoU损失 + 优化超参数**
