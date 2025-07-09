# 模型测试脚本使用说明

## 概述
`test_all_models.py` 现在支持通过命令行参数选择测试 best.pt 或 last.pt 模型，并可以自定义置信度和IoU阈值。

## 功能特性
- ✅ 支持选择 best.pt 或 last.pt 模型
- ✅ 支持选择不同的训练文件夹（train200epoch, train300epoch等）
- ✅ 自定义置信度阈值
- ✅ 自定义IoU阈值
- ✅ 自动生成带训练文件夹名和时间戳的输出目录
- ✅ NO-Safety Vest 类别召回率专门跟踪
- ✅ 详细的汇总报告和JSON结果
- ✅ 自动检测可用的训练文件夹

## 使用方法

### 基本用法

#### 列出可用的训练文件夹
```bash
python test_all_models.py --list-folders
```

#### 测试所有 best.pt 模型（默认train200epoch文件夹）
```bash
python test_all_models.py
```

#### 测试所有 last.pt 模型
```bash
python test_all_models.py --model-type last
```

#### 测试指定训练文件夹的模型
```bash
# 测试train300epoch文件夹下的best模型
python test_all_models.py --model-type best --train-folder train300epoch

# 测试train300epoch文件夹下的last模型
python test_all_models.py --model-type last --train-folder train300epoch
```

### 高级用法

#### 自定义置信度阈值
```bash
# 使用较高的置信度阈值（0.25）测试 best 模型
python test_all_models.py --model-type best --conf-thres 0.25

# 使用较高的置信度阈值（0.5）测试 last 模型
python test_all_models.py --model-type last --conf-thres 0.5
```

#### 自定义IoU阈值
```bash
# 使用较严格的IoU阈值（0.7）
python test_all_models.py --model-type best --iou-thres 0.7
```

#### 组合参数
```bash
# 同时自定义训练文件夹、模型类型、置信度和IoU阈值
python test_all_models.py --train-folder train300epoch --model-type last --conf-thres 0.25 --iou-thres 0.7

# 测试train300epoch文件夹下的best模型，使用高置信度阈值
python test_all_models.py --train-folder train300epoch --model-type best --conf-thres 0.5
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-type` | str | `best` | 选择模型类型：`best` 或 `last` |
| `--train-folder` | str | `train200epoch` | 训练文件夹名称，如 `train200epoch`, `train300epoch` |
| `--conf-thres` | float | `0.001` | 置信度阈值（0.0-1.0） |
| `--iou-thres` | float | `0.6` | IoU阈值（0.0-1.0） |
| `--list-folders` | flag | - | 列出所有可用的训练文件夹并退出 |

## 输出目录结构

### Train200epoch Best模型测试
```
runs/train200epoch_test_best_20250629_143022/
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
    ├── results.csv
    ├── labels/
    └── ...
```

### Train300epoch Last模型测试
```
runs/train300epoch_test_last_20250629_143022/
├── summary_report.txt          # 汇总报告
├── detailed_results.json       # 详细JSON结果
├── error_images/               # 预测错误的图片
└── ...                         # 各模型的详细结果
```

## 输出示例

### 运行时输出
```
开始测试 train300epoch 文件夹下的所有best.pt模型...
找到 5 个best模型:
  - yolov5s-ghost_1_: runs/train300epoch/yolov5s-ghost_1_/weights/best.pt
  - yolov5s-ghost_2_: runs/train300epoch/yolov5s-ghost_2_/weights/best.pt
  - yolov5s-ghost_3_: runs/train300epoch/yolov5s-ghost_3_/weights/best.pt
  - yolov5s_: runs/train300epoch/yolov5s_/weights/best.pt
  - yolov5s-ghost_123_: runs/train300epoch/yolov5s-ghost_123_/weights/best.pt

训练文件夹: runs/train300epoch
输出目录: runs/train300epoch_test_best_20250629_161302
置信度阈值: 0.25
IoU阈值: 0.6

[1/5] 测试模型: yolov5s-ghost_1_
正在测试模型: yolov5s-ghost_1_
模型 yolov5s-ghost_1_ 测试完成
  整体性能指标: Precision=0.839, Recall=0.840, mAP@0.5=0.867, mAP@0.5:0.95=0.535
  NO-Safety Vest召回率: 0.8090
  NO-Safety Vest其他指标: Precision=0.8090, mAP@0.5=0.8310, mAP@0.5:0.95=0.4420
```

### 汇总报告示例
```
====================================================================================================
YOLOv5 BEST模型测试汇总报告
====================================================================================================
测试时间: 2025-06-29 16:13:50
测试数据集: data/SafetyVests.v6/valid
训练文件夹: runs/train300epoch
模型类型: best.pt
模型数量: 5

整体模型性能对比:
----------------------------------------------------------------------------------------------------
模型名称              Precision    Recall       mAP@0.5      mAP@0.5:0.95   
----------------------------------------------------------------------------------------------------
yolov5s-ghost_1_     0.8790       0.8140       0.8660       0.4920         
yolov5s-ghost_2_     0.8650       0.8200       0.8580       0.4850         
yolov5s-ghost_3_     0.8720       0.8180       0.8620       0.4890         

NO-Safety Vest 类别性能对比:
----------------------------------------------------------------------------------------------------
模型名称              Precision    Recall       mAP@0.5      mAP@0.5:0.95   
----------------------------------------------------------------------------------------------------
yolov5s-ghost_1_     0.8460       0.7980       0.8340       0.4030         
yolov5s-ghost_2_     0.8320       0.8100       0.8280       0.3950         
yolov5s-ghost_3_     0.8400       0.8050       0.8310       0.4010         

最佳模型分析:
--------------------------------------------------
整体最佳模型 (基于mAP@0.5): yolov5s-ghost_1_
最佳整体mAP@0.5: 0.8660

NO-Safety Vest召回率最佳模型: yolov5s-ghost_2_
最佳NO-Safety Vest召回率: 0.8100
```

## 实际测试结果示例

### Best模型测试结果（置信度阈值=0.001）
```
整体最佳模型 (基于mAP@0.5): yolov5s-ghost_2_ (mAP@0.5: 0.9050)
NO-Safety Vest召回率最佳模型: yolov5s-ghost_2_ (召回率: 0.8280)
```

### Last模型测试结果（置信度阈值=0.25）
```
整体最佳模型 (基于mAP@0.5): yolov5s_ (mAP@0.5: 0.8890)
NO-Safety Vest召回率最佳模型: yolov5s-ghost_2_ (召回率: 0.8420)
```

### 置信度阈值对比分析
| 模型类型 | 置信度阈值 | 最佳整体mAP@0.5 | 最佳NO-Safety Vest召回率 |
|----------|------------|-----------------|-------------------------|
| best.pt  | 0.001      | 0.9050          | 0.8280                  |
| last.pt  | 0.25       | 0.8890          | 0.8420                  |

**观察结果**：
- 使用较高置信度阈值（0.25）时，last模型在NO-Safety Vest召回率上表现更好
- best模型在低置信度阈值下整体mAP表现更优
- 不同置信度阈值下的最佳模型可能不同

## 注意事项

1. **置信度阈值选择**：
   - 验证模式建议使用低阈值（0.001）以评估完整的召回率
   - 实际应用建议使用较高阈值（0.25-0.5）以获得可靠检测
   - 置信度阈值显著影响检测结果和性能指标

2. **模型类型选择**：
   - `best.pt`：训练过程中验证集上表现最好的模型
   - `last.pt`：训练结束时的最终模型
   - 两种模型在不同置信度阈值下可能表现不同

3. **输出目录**：
   - 每次运行都会创建新的时间戳目录
   - 目录名格式：`test_{model_type}_{timestamp}`
   - 避免结果覆盖，便于对比分析

4. **性能对比**：
   - 汇总报告提供整体性能和NO-Safety Vest专门性能对比
   - 可以同时识别整体最佳和特定类别最佳的模型
   - 支持跨不同配置的性能对比分析

5. **错误图片分析**：
   - 自动保存预测错误的图片到 `error_images/` 目录
   - 按模型分类组织，便于错误分析
   - 置信度阈值越高，错误图片数量通常越少
