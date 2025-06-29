# YOLOv5 模型配置说明

## 基础模型系列

### 原始基线模型

- **yolov5s.yaml**: 原始YOLOv5s基线方案，未做任何优化

### Ghost轻量化系列

- **yolov5s-ghost_1.yaml**: 仅添加Ghost模块（GhostConv和C3Ghost）
- **yolov5s-ghost_2.yaml**: 仅添加CA注意力机制
- **yolov5s-ghost_12.yaml**: 同时包含Ghost模块和CA注意力机制
- **yolov5s-ghost.yaml**: 最终推荐方案，目前和yolov5s-ghost_12.yaml相同

## 🆕 全面优化系列

### 高级优化模型

- **yolov5s-enhanced.yaml**: 全面优化版本
  - SE注意力机制替代CA注意力
  - 改进的CSP结构（EnhancedCSP）
  - 可学习上采样（LearnableUpsample）
  - 改进的检测头（EnhancedDetect）

## 训练参数说明

### 损失函数选择

```bash
--box-loss ciou # CIOU损失函数（默认）
--box-loss wiou # WIOU损失函数（推荐小目标检测）
```

### 超参数配置

```bash
--hyp data/hyps/hyp.recommend.yaml # 推荐的数据增强超参数（已设置为默认）
```

## 模型选择建议

| 使用场景         | 推荐模型                 | 特点           |
| ---------------- | ------------------------ | -------------- |
| **基线对比**     | yolov5s.yaml             | 原始性能基准   |
| **轻量化部署**   | yolov5s-ghost_1.yaml     | 参数少，速度快 |
| **精度优先**     | yolov5s-ghost_2.yaml     | 注意力增强     |
| **平衡性能**     | yolov5s-ghost_12.yaml    | 轻量化+注意力  |
| **高级优化**     | yolov5s-enhanced.yaml    | 多种先进技术   |
| **安全背心检测** | yolov5s-safety-vest.yaml | 专门优化       |

## 优化技术对比

| 技术         | Ghost_1 | Ghost_2 | Ghost_12 | Enhanced | Safety-Vest |
| ------------ | ------- | ------- | -------- | -------- | ----------- |
| Ghost模块    | ✅      | ❌      | ✅       | ❌       | ❌          |
| CA注意力     | ❌      | ✅      | ✅       | ❌       | ✅          |
| SE注意力     | ❌      | ❌      | ❌       | ✅       | ✅          |
| 改进CSP      | ❌      | ❌      | ❌       | ✅       | ❌          |
| 可学习上采样 | ❌      | ❌      | ❌       | ✅       | ✅          |
| 改进检测头   | ❌      | ❌      | ❌       | ✅       | ✅          |
| 小目标优化   | ❌      | ❌      | ❌       | ❌       | ✅          |
| 专用锚框     | ❌      | ❌      | ❌       | ❌       | ✅          |

## 使用示例

### 基础训练

```bash
# 原始基线
python train.py --cfg models/yolov5s.yaml --data data.yaml

# Ghost轻量化
python train.py --cfg models/yolov5s-ghost_1.yaml --data data.yaml

# 完整优化
python train.py --cfg models/yolov5s-ghost_12.yaml --data data.yaml --box-loss wiou
```

### 高级训练

```bash
# 全面优化模型
python train.py --cfg models/yolov5s-enhanced.yaml --data data.yaml --box-loss wiou

# 安全背心专用（推荐）
python train.py --cfg models/yolov5s-safety-vest.yaml --data data/SafetyVests.v6/data.yaml --box-loss wiou
```

### 锚框优化

```bash
# 为安全背心数据集计算最优锚框
python utils/anchor_optimization.py --data data/SafetyVests.v6/data.yaml --img 640
```
