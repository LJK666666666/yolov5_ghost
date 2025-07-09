#!/bin/bash

# 使用safety-helmet-vest数据集训练多个YOLOv5模型
# 基于您提供的训练命令，修改了数据集路径和项目路径，批次大小改为32

echo "🚀 开始训练 safety-helmet-vest 数据集上的多个YOLOv5模型"
echo "数据集: data/safety-helmet-vest/data.yaml"
echo "输出目录: runs/safety_helmet_train300epoch"
echo "批次大小: 32"
echo "训练轮数: 300"

# 检查数据集是否存在
if [ ! -f "data/safety-helmet-vest/data.yaml" ]; then
    echo "❌ 错误: 数据集配置文件不存在!"
    echo "请先运行: python download_safety_helmet_dataset.py"
    exit 1
fi

# 记录开始时间
start_time=$(date +%s)

echo ""
echo "📊 训练计划:"
echo "1. YOLOv5s Baseline"
echo "2. YOLOv5s-Ghost_123 + WIoU Loss"  
echo "3. YOLOv5s-Ghost_1"
echo "4. YOLOv5s-Ghost_2"
echo "5. YOLOv5s + WIoU Loss"

# 训练1: YOLOv5s Baseline
echo ""
echo "🔥 [1/5] 开始训练: YOLOv5s Baseline"
python train.py \
    --cfg models/yolov5s.yaml \
    --data data/safety-helmet-vest/data.yaml \
    --weights yolov5s.pt \
    --project runs/safety_helmet_train300epoch \
    --name yolov5s_ \
    --epochs 300 \
    --patience 100 \
    --batch-size 32

if [ $? -eq 0 ]; then
    echo "✅ YOLOv5s Baseline 训练完成"
else
    echo "❌ YOLOv5s Baseline 训练失败"
fi

# 训练2: YOLOv5s-Ghost_123 + WIoU Loss
echo ""
echo "🔥 [2/5] 开始训练: YOLOv5s-Ghost_123 + WIoU Loss"
python train.py \
    --cfg models/yolov5s-ghost_12.yaml \
    --data data/safety-helmet-vest/data.yaml \
    --weights yolov5s.pt \
    --project runs/safety_helmet_train300epoch \
    --name yolov5s-ghost_123_ \
    --box-loss wiou \
    --epochs 300 \
    --patience 100 \
    --batch-size 32

if [ $? -eq 0 ]; then
    echo "✅ YOLOv5s-Ghost_123 + WIoU 训练完成"
else
    echo "❌ YOLOv5s-Ghost_123 + WIoU 训练失败"
fi

# 训练3: YOLOv5s-Ghost_1
echo ""
echo "🔥 [3/5] 开始训练: YOLOv5s-Ghost_1"
python train.py \
    --cfg models/yolov5s-ghost_1.yaml \
    --data data/safety-helmet-vest/data.yaml \
    --weights yolov5s.pt \
    --project runs/safety_helmet_train300epoch \
    --name yolov5s-ghost_1_ \
    --epochs 300 \
    --patience 100 \
    --batch-size 32

if [ $? -eq 0 ]; then
    echo "✅ YOLOv5s-Ghost_1 训练完成"
else
    echo "❌ YOLOv5s-Ghost_1 训练失败"
fi

# 训练4: YOLOv5s-Ghost_2
echo ""
echo "🔥 [4/5] 开始训练: YOLOv5s-Ghost_2"
python train.py \
    --cfg models/yolov5s-ghost_2.yaml \
    --data data/safety-helmet-vest/data.yaml \
    --weights yolov5s.pt \
    --project runs/safety_helmet_train300epoch \
    --name yolov5s-ghost_2_ \
    --epochs 300 \
    --patience 100 \
    --batch-size 32

if [ $? -eq 0 ]; then
    echo "✅ YOLOv5s-Ghost_2 训练完成"
else
    echo "❌ YOLOv5s-Ghost_2 训练失败"
fi

# 训练5: YOLOv5s + WIoU Loss
echo ""
echo "🔥 [5/5] 开始训练: YOLOv5s + WIoU Loss"
python train.py \
    --cfg models/yolov5s.yaml \
    --data data/safety-helmet-vest/data.yaml \
    --weights yolov5s.pt \
    --project runs/safety_helmet_train300epoch \
    --name yolov5s-ghost_3_ \
    --box-loss wiou \
    --epochs 300 \
    --patience 100 \
    --batch-size 32

if [ $? -eq 0 ]; then
    echo "✅ YOLOv5s + WIoU 训练完成"
else
    echo "❌ YOLOv5s + WIoU 训练失败"
fi

# 计算总训练时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo ""
echo "🎯 所有训练任务完成!"
echo "总训练时间: ${hours}小时 ${minutes}分钟"
echo "训练结果保存在: runs/safety_helmet_train300epoch/"

# 显示训练结果目录
echo ""
echo "📁 训练结果目录:"
if [ -d "runs/safety_helmet_train300epoch" ]; then
    ls -la runs/safety_helmet_train300epoch/
else
    echo "训练结果目录不存在"
fi
