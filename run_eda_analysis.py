#!/usr/bin/env python3
"""SafetyVests.v6 数据集探索性数据分析脚本 快速运行主要分析并生成报告."""

import glob
import os
from collections import Counter

import numpy as np
import yaml
from PIL import Image


def count_files(path, extensions=["*.jpg", "*.jpeg", "*.png"]):
    """统计指定路径下的文件数量."""
    count = 0
    for ext in extensions:
        count += len(glob.glob(os.path.join(path, ext)))
    return count


def analyze_image_sizes(image_path, sample_size=200):
    """分析图像尺寸分布."""
    image_files = glob.glob(os.path.join(image_path, "*.jpg"))

    # 如果图像太多，随机采样
    if len(image_files) > sample_size:
        image_files = np.random.choice(image_files, sample_size, replace=False)

    widths, heights, ratios = [], [], []

    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)
        except Exception as e:
            print(f"无法读取图像 {img_file}: {e}")

    return widths, heights, ratios


def parse_yolo_labels(labels_path):
    """解析YOLO格式标签文件."""
    all_annotations = []
    class_counts = Counter()
    bbox_areas = []
    bbox_widths = []
    bbox_heights = []
    objects_per_image = []

    label_files = glob.glob(os.path.join(labels_path, "*.txt"))

    for label_file in label_files:
        with open(label_file) as f:
            lines = f.readlines()

        objects_count = 0
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    class_counts[class_id] += 1
                    bbox_areas.append(width * height)
                    bbox_widths.append(width)
                    bbox_heights.append(height)
                    objects_count += 1

                    all_annotations.append(
                        {
                            "class_id": class_id,
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "area": width * height,
                        }
                    )

        objects_per_image.append(objects_count)

    return all_annotations, class_counts, bbox_areas, bbox_widths, bbox_heights, objects_per_image


def check_data_quality(images_path, labels_path):
    """检查数据质量问题."""
    issues = []

    image_files = set(
        [os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith((".jpg", ".jpeg", ".png"))]
    )
    label_files = set([os.path.splitext(f)[0] for f in os.listdir(labels_path) if f.endswith(".txt")])

    # 检查图像和标签文件匹配
    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    if missing_labels:
        issues.append(f"缺少标签文件的图像: {len(missing_labels)} 个")
    if missing_images:
        issues.append(f"缺少图像文件的标签: {len(missing_images)} 个")

    # 检查空标签文件
    empty_labels = 0
    invalid_annotations = 0

    for label_file in glob.glob(os.path.join(labels_path, "*.txt")):
        with open(label_file) as f:
            lines = f.readlines()

        if not lines or all(not line.strip() for line in lines):
            empty_labels += 1

        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_annotations += 1
                else:
                    try:
                        int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        # 检查坐标范围
                        if not all(0 <= coord <= 1 for coord in coords):
                            invalid_annotations += 1
                    except ValueError:
                        invalid_annotations += 1

    if empty_labels > 0:
        issues.append(f"空标签文件: {empty_labels} 个")
    if invalid_annotations > 0:
        issues.append(f"无效标注: {invalid_annotations} 个")

    return issues


def main():
    """主分析函数."""
    print("SafetyVests.v6 数据集探索性数据分析")
    print("=" * 60)

    # 数据集路径配置
    dataset_path = "data/SafetyVests.v6"
    train_images_path = os.path.join(dataset_path, "train/images")
    train_labels_path = os.path.join(dataset_path, "train/labels")
    valid_images_path = os.path.join(dataset_path, "valid/images")
    valid_labels_path = os.path.join(dataset_path, "valid/labels")
    test_images_path = os.path.join(dataset_path, "test/images")
    test_labels_path = os.path.join(dataset_path, "test/labels")

    # 读取数据集配置文件
    with open(os.path.join(dataset_path, "data.yaml")) as f:
        data_config = yaml.safe_load(f)

    class_names = data_config["names"]

    print("\n1. 数据集基本信息:")
    print(f"   类别数量: {data_config['nc']}")
    print(f"   类别名称: {class_names}")

    # 统计各分割的图像数量
    train_img_count = count_files(train_images_path)
    valid_img_count = count_files(valid_images_path)
    test_img_count = count_files(test_images_path)
    total_images = train_img_count + valid_img_count + test_img_count

    print("\n2. 数据分布:")
    print(f"   训练集: {train_img_count:,} 图像 ({train_img_count / total_images * 100:.1f}%)")
    print(f"   验证集: {valid_img_count:,} 图像 ({valid_img_count / total_images * 100:.1f}%)")
    print(f"   测试集: {test_img_count:,} 图像 ({test_img_count / total_images * 100:.1f}%)")
    print(f"   总计: {total_images:,} 图像")

    # 分析图像尺寸
    print("\n3. 图像尺寸分析 (基于训练集样本):")
    train_widths, train_heights, train_ratios = analyze_image_sizes(train_images_path)

    if train_widths:
        print(f"   平均宽度: {np.mean(train_widths):.0f} ± {np.std(train_widths):.0f} 像素")
        print(f"   平均高度: {np.mean(train_heights):.0f} ± {np.std(train_heights):.0f} 像素")
        print(f"   宽度范围: {min(train_widths)} - {max(train_widths)} 像素")
        print(f"   高度范围: {min(train_heights)} - {max(train_heights)} 像素")
        print(f"   平均宽高比: {np.mean(train_ratios):.2f} ± {np.std(train_ratios):.2f}")

    # 分析标注
    print("\n4. 标注分析:")
    (
        train_annotations,
        train_class_counts,
        train_bbox_areas,
        train_bbox_widths,
        train_bbox_heights,
        train_objects_per_image,
    ) = parse_yolo_labels(train_labels_path)

    total_annotations = sum(train_class_counts.values())
    print(f"   训练集总标注数量: {total_annotations:,}")
    print(f"   平均每张图像的目标数量: {np.mean(train_objects_per_image):.2f}")

    print("\n5. 类别分布:")
    for class_id, count in train_class_counts.items():
        percentage = count / total_annotations * 100
        print(f"   {class_names[class_id]}: {count:,} ({percentage:.1f}%)")

    # 边界框统计
    if train_bbox_areas:
        print("\n6. 边界框特征 (归一化坐标):")
        print(f"   平均面积: {np.mean(train_bbox_areas):.4f} ± {np.std(train_bbox_areas):.4f}")
        print(f"   平均宽度: {np.mean(train_bbox_widths):.4f} ± {np.std(train_bbox_widths):.4f}")
        print(f"   平均高度: {np.mean(train_bbox_heights):.4f} ± {np.std(train_bbox_heights):.4f}")

        bbox_ratios = [w / h for w, h in zip(train_bbox_widths, train_bbox_heights)]
        print(f"   平均宽高比: {np.mean(bbox_ratios):.2f} ± {np.std(bbox_ratios):.2f}")

    # 数据质量检查
    print("\n7. 数据质量检查:")

    print("   训练集:")
    train_issues = check_data_quality(train_images_path, train_labels_path)
    if train_issues:
        for issue in train_issues:
            print(f"     - {issue}")
    else:
        print("     ✓ 未发现问题")

    print("   验证集:")
    valid_issues = check_data_quality(valid_images_path, valid_labels_path)
    if valid_issues:
        for issue in valid_issues:
            print(f"     - {issue}")
    else:
        print("     ✓ 未发现问题")

    print("   测试集:")
    test_issues = check_data_quality(test_images_path, test_labels_path)
    if test_issues:
        for issue in test_issues:
            print(f"     - {issue}")
    else:
        print("     ✓ 未发现问题")

    # 总结和建议
    print("\n8. 总结和建议:")
    print("   ✓ 数据集规模适中，适合训练YOLO模型")

    if len(train_class_counts) == 2:
        class_balance = (
            abs(list(train_class_counts.values())[0] - list(train_class_counts.values())[1]) / total_annotations
        )
        if class_balance < 0.3:
            print("   ✓ 类别分布相对均衡")
        else:
            print("   ⚠ 类别分布不均衡，建议使用数据增强或权重平衡")

    if train_widths and (max(train_widths) - min(train_widths)) > 500:
        print("   ⚠ 图像尺寸变化较大，建议使用多尺度训练")
    else:
        print("   ✓ 图像尺寸相对一致")

    print("   ✓ 建议使用数据增强技术提高模型泛化能力")
    print("   ✓ 可以考虑使用预训练模型进行迁移学习")

    print("\n" + "=" * 60)
    print("分析完成！详细的可视化分析请运行 SafetyVests_EDA.ipynb")


if __name__ == "__main__":
    main()
