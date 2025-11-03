yolov5s.yaml表示基线方案
1表示添加ghost模块（GhostConv和C3Ghost）
2表示添加CA注意力机制
se表示添加SE注意力机制（Squeeze-and-Excitation）
moe表示添加MoE架构（Mixture of Experts）
yolov5s-ghost.yaml表示最终方案，目前和yolov5s-ghost_12.yaml相同
yolov5s-se.yaml表示集成SE注意力机制的方案
yolov5s-moe.yaml表示集成MoE架构的方案
--box-loss wiou启用WIOU损失函数，默认为CIOU损失函数
--hyp data\hyps\hyp.recommend.yaml启用推荐的数据增强超参数（已设置成默认）

## 模型配置说明

### 基础模型

- **yolov5s.yaml**: 原始YOLOv5s基线模型

### 优化模块

- **Ghost模块**: 通过Ghost卷积减少参数量和计算量，保持检测精度
- **CA注意力机制**: 坐标注意力机制，捕获跨通道信息和位置相关信息
- **SE注意力机制**: 通道注意力机制，学习通道间的相互依赖关系
- **MoE架构**: 混合专家架构，通过多个专家网络和门控机制提升模型容量

### 集成方案

- **yolov5s-ghost.yaml**: 集成Ghost模块的轻量化方案
- **yolov5s-se-conservative.yaml**: 保守SE策略（参数增加0.38%）
- **yolov5s-se.yaml**: 平衡SE策略（参数增加1.28%）
- **yolov5s-se-aggressive.yaml**: 激进SE策略（参数增加1.28%）
- **yolov5s-moe-lite.yaml**: 轻量级MoE架构（参数增加148%）
- **yolov5s-moe.yaml**: 完整MoE架构（更多专家）
- **yolov5s-adaptive-moe.yaml**: 自适应MoE架构（智能专家选择）

### SE注意力机制特点

1. **轻量级**: 参数量很少，计算开销小
2. **即插即用**: 可以轻松集成到现有网络架构中
3. **性能提升**: 能够显著提升模型的表征能力
4. **通道注意力**: 专注于"什么"特征是重要的

### MoE架构特点

1. **稀疏激活**: 每次只激活少数专家，保持计算效率
2. **高容量**: 可以有很多专家，大幅提升模型容量
3. **专业化**: 不同专家学习处理不同类型的特征
4. **可扩展**: 容易扩展到更多专家而不线性增加计算量

### SE策略对比

| 策略     | 参数增加 | SE模块数 | 适用场景         |
| -------- | -------- | -------- | ---------------- |
| 保守策略 | 0.38%    | 78       | 移动端、边缘设备 |
| 平衡策略 | 1.28%    | 160      | 通用场景、服务器 |
| 激进策略 | 1.28%    | 168      | 高精度需求       |

### MoE策略对比

| 策略      | 参数增加 | MoE模块数 | 专家数量 | 适用场景   |
| --------- | -------- | --------- | -------- | ---------- |
| 轻量级MoE | 148%     | 24        | 4-8个    | 高容量需求 |
| 完整MoE   | 200%+    | 40+       | 4-8个    | 最大容量   |
| 自适应MoE | 150%+    | 20+       | 6个      | 智能选择   |

### SE位置选择原理

1. **下采样后**: 重新校准通道重要性，适应新特征尺度
2. **特征融合前后**: 优化融合特征，增强多尺度检测
3. **深层特征**: SE在高维特征空间效果更明显
4. **计算效率**: 避免在高分辨率特征图上使用SE

### 使用方法

```bash
# 保守SE策略（资源受限）
python train.py --cfg models/yolov5s-se-conservative.yaml --data data/coco.yaml

# 平衡SE策略（推荐）
python train.py --cfg models/yolov5s-se.yaml --data data/coco.yaml

# 激进SE策略（精度优先）
python train.py --cfg models/yolov5s-se-aggressive.yaml --data data/coco.yaml

# 使用WIoU损失函数训练
python train.py --cfg models/yolov5s-se.yaml --data data/coco.yaml --box-loss wiou

# 使用推荐超参数训练
python train.py --cfg models/yolov5s-se.yaml --data data/coco.yaml --hyp data/hyps/hyp.recommend.yaml

# MoE架构训练
# 轻量级MoE（推荐）
python train.py --cfg models/yolov5s-moe-lite.yaml --data data/coco.yaml

# 完整MoE（高容量）
python train.py --cfg models/yolov5s-moe.yaml --data data/coco.yaml

# 自适应MoE（智能选择）
python train.py --cfg models/yolov5s-adaptive-moe.yaml --data data/coco.yaml
```
