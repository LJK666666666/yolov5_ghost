#!/usr/bin/env python3
"""简化的数据分析脚本."""

import json
import os
from collections import Counter

import cv2
import numpy as np
import yaml


def parse_yolo_label(label_file, img_width, img_height):
    """解析YOLO格式标签文件."""
    annotations = []
    if os.path.exists(label_file):
        with open(label_file) as f:
            lines = f.readlines()

        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    annotations.append(
                        {
                            "class_id": class_id,
                            "bbox": [x1, y1, x2, y2],
                            "size": [width, height],
                            "area": width * height,
                        }
                    )

    return annotations


def calculate_overlap_ratio(small_box, large_box):
    """计算小框在大框中的重叠比例."""
    x1 = max(small_box[0], large_box[0])
    y1 = max(small_box[1], large_box[1])
    x2 = min(small_box[2], large_box[2])
    y2 = min(small_box[3], large_box[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])

    return intersection / small_area if small_area > 0 else 0.0


def analyze_equipment_status(person_bbox, helmets, safety_vests, overlap_threshold=0.3):
    """分析单个person的装备穿戴情况."""
    has_helmet = False
    has_vest = False

    # 检查helmet
    for helmet in helmets:
        overlap_ratio = calculate_overlap_ratio(helmet["bbox"], person_bbox)
        if overlap_ratio >= overlap_threshold:
            has_helmet = True
            break

    # 检查safety_vest
    for vest in safety_vests:
        overlap_ratio = calculate_overlap_ratio(vest["bbox"], person_bbox)
        if overlap_ratio >= overlap_threshold:
            has_vest = True
            break

    # 确定装备状态
    if has_helmet and has_vest:
        return "fully_equipped"
    elif has_helmet:
        return "helmet_only"
    elif has_vest:
        return "vest_only"
    else:
        return "no_equipment"


def main():
    print("开始数据分析...")

    # 数据集路径
    data_dir = "data/railroad-worker-detection"

    # 检查路径
    if not os.path.exists(data_dir):
        print(f"错误: 数据集路径不存在 {data_dir}")
        return

    # 读取类别配置
    config_file = os.path.join(data_dir, "data.yaml")
    with open(config_file) as f:
        data_config = yaml.safe_load(f)

    class_names = data_config["names"]
    print(f"类别: {class_names}")

    # 分析训练数据
    images_dir = os.path.join(data_dir, "train/images")
    labels_dir = os.path.join(data_dir, "train/labels")

    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在 {images_dir}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    print(f"找到 {len(image_files)} 张图像")

    # 统计数据
    equipment_status_counts = Counter()
    person_sizes = []
    image_sizes = []

    # 处理前100张图像进行快速分析
    sample_size = min(100, len(image_files))
    print(f"分析前 {sample_size} 张图像...")

    for i, img_file in enumerate(image_files[:sample_size]):
        if i % 20 == 0:
            print(f"处理进度: {i}/{sample_size}")

        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            continue

        img_height, img_width = image.shape[:2]
        image_sizes.append((img_width, img_height))

        # 解析标签
        annotations = parse_yolo_label(label_path, img_width, img_height)

        # 按类别分组
        persons = [ann for ann in annotations if class_names[ann["class_id"]] == "person"]
        helmets = [ann for ann in annotations if class_names[ann["class_id"]] == "helmet"]
        safety_vests = [ann for ann in annotations if class_names[ann["class_id"]] == "safety_vest"]

        # 分析每个person的装备状态
        for person in persons:
            equipment_status = analyze_equipment_status(person["bbox"], helmets, safety_vests)
            equipment_status_counts[equipment_status] += 1
            person_sizes.append(person["size"])

    # 输出统计结果
    print("\n" + "=" * 50)
    print("数据分析结果")
    print("=" * 50)

    print(f"\n图像统计 (样本: {len(image_sizes)}):")
    if image_sizes:
        avg_width = np.mean([size[0] for size in image_sizes])
        avg_height = np.mean([size[1] for size in image_sizes])
        print(f"  平均尺寸: {avg_width:.0f} x {avg_height:.0f}")

    print(f"\n装备状态分布 (样本: {sum(equipment_status_counts.values())}):")
    total_persons = sum(equipment_status_counts.values())
    for status, count in equipment_status_counts.items():
        percentage = count / total_persons * 100 if total_persons > 0 else 0
        print(f"  {status}: {count} ({percentage:.1f}%)")

    print(f"\n人员尺寸统计 (样本: {len(person_sizes)}):")
    if person_sizes:
        avg_person_width = np.mean([size[0] for size in person_sizes])
        avg_person_height = np.mean([size[1] for size in person_sizes])
        print(f"  平均人员尺寸: {avg_person_width:.0f} x {avg_person_height:.0f}")

        min_person_width = np.min([size[0] for size in person_sizes])
        max_person_width = np.max([size[0] for size in person_sizes])
        min_person_height = np.min([size[1] for size in person_sizes])
        max_person_height = np.max([size[1] for size in person_sizes])

        print(f"  人员宽度范围: {min_person_width:.0f} - {max_person_width:.0f}")
        print(f"  人员高度范围: {min_person_height:.0f} - {max_person_height:.0f}")

    # 保存结果
    results = {
        "sample_size": sample_size,
        "total_images": len(image_files),
        "equipment_status_distribution": dict(equipment_status_counts),
        "average_image_size": [avg_width, avg_height] if image_sizes else [0, 0],
        "average_person_size": [avg_person_width, avg_person_height] if person_sizes else [0, 0],
        "person_size_range": {
            "width": [min_person_width, max_person_width] if person_sizes else [0, 0],
            "height": [min_person_height, max_person_height] if person_sizes else [0, 0],
        },
    }

    os.makedirs("vae_analysis_results", exist_ok=True)
    with open("vae_analysis_results/quick_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n分析完成! 结果保存在: vae_analysis_results/quick_analysis.json")

    # 基于分析结果提供VAE设计建议
    print("\n" + "=" * 50)
    print("VAE设计建议")
    print("=" * 50)

    if person_sizes:
        print(f"1. 人员区域标准化尺寸建议: {int(avg_person_width)} x {int(avg_person_height)}")

    print("2. 需要重点增强的类别:")
    if equipment_status_counts:
        sorted_status = sorted(equipment_status_counts.items(), key=lambda x: x[1])
        for status, count in sorted_status[:2]:  # 显示最少的两个类别
            print(f"   - {status}: {count} 个样本")

    print("3. 建议的VAE架构:")
    print("   - 背景VAE: 学习工作环境背景")
    print("   - 条件人员VAE: 根据装备状态生成人员")
    print("   - 建议潜在空间维度: 128-256")


if __name__ == "__main__":
    main()
