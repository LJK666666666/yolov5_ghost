# YOLOv5 模型分析工具使用指南

## 📋 功能概述

`analyze_models.py` 是一个强大的模型分析工具，可以分析指定文件夹下所有 `best.pt` 模型的：

- 参数量（总参数、可训练参数）
- 计算复杂度（GFLOPs）
- 推理速度（推理时间、FPS）
- 模型文件大小

## 🚀 基本使用

### 1. 基本命令

```bash
# 分析默认文件夹 runs/train200epoch
python analyze_models.py

# 分析指定文件夹
python analyze_models.py --folder runs/sv6_train1000epoch_

# 使用简写参数
python analyze_models.py -f runs/sv6_train1000epoch_
```

### 2. 指定输出路径

```bash
# 将结果保存到指定文件夹
python analyze_models.py --folder runs/sv6_train1000epoch_ --output analysis_results

# 使用简写参数
python analyze_models.py -f runs/sv6_train1000epoch_ -o analysis_results
```

### 3. 设备选择

```bash
# 使用CPU进行分析
python analyze_models.py --folder runs/sv6_train1000epoch_ --device cpu

# 使用指定GPU
python analyze_models.py --folder runs/sv6_train1000epoch_ --device cuda:0

# 自动选择设备（默认）
python analyze_models.py --folder runs/sv6_train1000epoch_
```

### 4. 自定义测试参数

```bash
# 自定义输入尺寸和测试次数
python analyze_models.py \
  --folder runs/sv6_train1000epoch_ \
  --input-size 416 416 \
  --num-runs 50 \
  --device cuda:0
```

## 📊 输出文件

分析完成后会生成以下文件：

### 1. Excel 汇总表 (`模型分析汇总.xlsx`)

- 包含所有模型的关键指标
- 便于在Excel中进行进一步分析

### 2. CSV 汇总表 (`模型分析汇总.csv`)

- 与Excel相同的数据，CSV格式
- 便于程序化处理

### 3. 详细报告 (`模型分析详细报告.txt`)

- 包含统计信息和最佳模型推荐
- 人类可读的详细分析结果

### 4. JSON 详细结果 (`模型分析详细结果.json`)

- 包含所有原始数据
- 便于程序化处理和进一步分析

## 🎯 参数说明

| 参数           | 简写 | 默认值               | 说明                             |
| -------------- | ---- | -------------------- | -------------------------------- |
| `--folder`     | `-f` | `runs/train200epoch` | 要分析的训练文件夹路径           |
| `--output`     | `-o` | 与输入文件夹相同     | 输出结果的文件夹路径             |
| `--device`     | -    | 自动选择             | 使用的设备 (cpu/cuda:0/cuda:1等) |
| `--num-runs`   | -    | 100                  | 推理速度测试的运行次数           |
| `--input-size` | -    | 640 640              | 输入图像尺寸 (高度 宽度)         |

## 💡 使用示例

### 示例1：快速分析

```bash
# 快速分析sv6训练结果，使用默认参数
python analyze_models.py -f runs/sv6_train1000epoch_
```

### 示例2：详细分析

```bash
# 详细分析，自定义所有参数
python analyze_models.py \
  --folder runs/sv6_train1000epoch_ \
  --output detailed_analysis \
  --device cuda:0 \
  --input-size 640 640 \
  --num-runs 100
```

### 示例3：CPU性能测试

```bash
# 在CPU上测试模型性能，使用较小输入尺寸
python analyze_models.py \
  --folder runs/sv6_train1000epoch_ \
  --output cpu_analysis \
  --device cpu \
  --input-size 416 416 \
  --num-runs 20
```

### 示例4：移动端优化分析

```bash
# 模拟移动端环境，小尺寸输入，少量测试
python analyze_models.py \
  --folder runs/sv6_train1000epoch_ \
  --output mobile_analysis \
  --device cpu \
  --input-size 320 320 \
  --num-runs 10
```

## 📈 结果解读

### 关键指标说明

- **参数量**：模型的总参数数量，影响模型大小和内存占用
- **GFLOPs**：十亿次浮点运算，衡量计算复杂度
- **推理时间**：单张图片的推理时间（毫秒）
- **FPS**：每秒处理的图片数量
- **文件大小**：模型文件的磁盘占用空间

### 性能权衡

- **参数量 ↓** = 模型更小，内存占用更少
- **GFLOPs ↓** = 计算量更少，推理更快
- **推理时间 ↓** = 实时性更好
- **FPS ↑** = 吞吐量更高

## ⚠️ 注意事项

1. **thop库依赖**：如果未安装thop库，GFLOPs计算将不可用

    ```bash
    pip install thop
    ```

2. **设备一致性**：建议在目标部署设备上进行性能测试

3. **测试次数**：增加`--num-runs`可以获得更准确的性能数据，但会增加测试时间

4. **输入尺寸**：不同输入尺寸会显著影响性能指标

## 🔧 故障排除

### 常见问题

1. **找不到模型文件**：确保文件夹路径正确，且包含`weights/best.pt`文件
2. **CUDA内存不足**：使用`--device cpu`或减少`--num-runs`
3. **thop计算失败**：某些自定义模块可能不被thop支持，这是正常现象

### 获取帮助

```bash
python analyze_models.py --help
```
