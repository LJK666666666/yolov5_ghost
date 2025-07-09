# 层次化安全装备检测分析脚本

## 概述

`hierarchical_safety_analysis.py` 是一个专门为安全装备检测任务设计的分析脚本，能够智能处理 `person`、`helmet` 和 `safety_vest` 之间的层次关系，避免传统分析方法中的误判问题。

## 主要特性

### 1. 智能层次化匹配
- **解决传统问题**：避免将person身上的safety_vest预测误判为False Positive
- **装备关联分析**：自动分析每个person的装备穿戴情况
- **重叠检测**：通过IoU和重叠比例判断装备是否属于特定person

### 2. 四种装备状态分类
- `fully_equipped`: 同时佩戴helmet和safety_vest
- `helmet_only`: 仅佩戴helmet
- `vest_only`: 仅佩戴safety_vest  
- `no_equipment`: 未佩戴任何装备

### 3. 多层次错误分析
- **Person检测错误**：Person级别的TP/FP/FN
- **装备状态错误**：装备穿戴状态的分类错误
- **独立组件错误**：不属于任何person的helmet/vest检测

## 使用方法

### 基本用法
```bash
python hierarchical_safety_analysis.py \
    --weights runs/rail_train300epoch/yolov5s_/weights/best.pt \
    --data data/railroad-worker-detection \
    --output hierarchical_analysis
```

### 参数说明
- `--weights`: 模型权重文件路径
- `--data`: 数据集根目录路径
- `--output`: 分析结果输出目录
- `--conf`: 置信度阈值 (默认: 0.1)
- `--overlap`: 装备重叠阈值 (默认: 0.3)

### 数据集要求
数据集必须包含以下三个类别：
- `safety_vest` (类别0)
- `helmet` (类别1)
- `person` (类别2)

## 输出结果

### 目录结构
```
hierarchical_analysis/
├── train/
│   ├── person_detection_errors/     # Person检测错误的图像
│   ├── equipment_status_errors/     # 装备状态错误的图像
│   ├── independent_components/      # 独立组件错误的图像
│   ├── correct_predictions/         # 正确预测的示例图像
│   ├── hierarchical_statistics.json # 详细统计信息
│   └── detailed_hierarchical_results.json # 详细结果
├── valid/
├── test/
└── overall_hierarchical_statistics.json # 总体统计
```

### 统计指标

#### Person检测性能
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)  
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)

#### 装备状态准确率
每种装备状态的分类准确率：
- `fully_equipped`: 完全装备状态的准确率
- `helmet_only`: 仅helmet状态的准确率
- `vest_only`: 仅vest状态的准确率
- `no_equipment`: 无装备状态的准确率

#### 组件检测统计
独立helmet和safety_vest的检测统计（不属于任何person的组件）

## 与传统分析的区别

### 传统分析问题
```python
# 传统方法的问题场景
真实标签: person (类别2) 在区域A
模型预测: safety_vest (类别0) 在区域A的子区域
结果: safety_vest被误判为False Positive，person被误判为False Negative
```

### 层次化分析解决方案
```python
# 层次化方法的处理
1. 检测到person在区域A
2. 检测到safety_vest在区域A的子区域
3. 计算重叠比例，判断safety_vest属于该person
4. 将person标记为"vest_only"状态
5. 正确评估：person检测正确，装备状态正确
```

## 核心算法

### 重叠比例计算
```python
overlap_ratio = intersection_area / small_box_area
```
当重叠比例 >= overlap_threshold 时，认为装备属于该person。

### 装备状态判断
```python
if has_helmet and has_vest:
    status = "fully_equipped"
elif has_helmet:
    status = "helmet_only"
elif has_vest:
    status = "vest_only"
else:
    status = "no_equipment"
```

## 可视化说明

生成的图像使用不同颜色标识：
- **绿色框**: 正确的person检测和装备状态
- **红色框**: 错误的person检测或装备状态
- **蓝色框**: 漏检的person
- **青色小框**: helmet (标记为"H")
- **紫色小框**: safety_vest (标记为"V")

## 适用场景

1. **安全监控系统**：工地、工厂等场所的安全装备合规检查
2. **模型性能评估**：更准确地评估安全装备检测模型的性能
3. **数据集质量分析**：发现标注中的潜在问题
4. **模型改进指导**：识别模型在不同装备状态下的表现差异

## 注意事项

1. **重叠阈值调整**：根据数据集特点调整`--overlap`参数
2. **置信度设置**：较低的置信度阈值有助于发现更多潜在问题
3. **类别顺序**：确保data.yaml中的类别顺序正确
4. **内存使用**：大数据集处理时注意内存使用情况

## 示例输出

```
层次化安全装备检测分析
==================================================
类别: ['safety_vest', 'helmet', 'person']

处理train...
处理进度: 0/1000
...
train处理完成:
  总图像数: 1000
  正确预测: 856
  准确率: 85.60%
  Person检测 - P: 0.923, R: 0.887, F1: 0.905
  装备状态准确率: {'fully_equipped': 0.892, 'helmet_only': 0.756, 'vest_only': 0.834, 'no_equipment': 0.901}
```
