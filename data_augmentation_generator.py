#!/usr/bin/env python3
"""
数据增强生成器
使用传统数据增强技术生成更多稀少类别样本.
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np


class DataAugmenter:
    """数据增强器."""

    def __init__(self):
        self.augmentation_methods = [
            self.horizontal_flip,
            self.brightness_adjustment,
            self.contrast_adjustment,
            self.gaussian_noise,
            self.rotation,
            self.scale_adjustment,
            self.color_shift,
            self.gaussian_blur,
        ]

    def horizontal_flip(self, image):
        """水平翻转."""
        return cv2.flip(image, 1)

    def brightness_adjustment(self, image):
        """亮度调整."""
        factor = random.uniform(0.7, 1.3)
        adjusted = image * factor
        return np.clip(adjusted, 0, 1)

    def contrast_adjustment(self, image):
        """对比度调整."""
        factor = random.uniform(0.8, 1.2)
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 1)

    def gaussian_noise(self, image):
        """高斯噪声."""
        noise = np.random.normal(0, 0.02, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)

    def rotation(self, image):
        """旋转."""
        angle = random.uniform(-10, 10)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    def scale_adjustment(self, image):
        """尺度调整."""
        scale = random.uniform(0.9, 1.1)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        if scale > 1:
            # 放大后裁剪
            resized = cv2.resize(image, (new_w, new_h))
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h : start_h + h, start_w : start_w + w]
        else:
            # 缩小后填充
            resized = cv2.resize(image, (new_w, new_h))
            padded = np.zeros_like(image)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            padded[start_h : start_h + new_h, start_w : start_w + new_w] = resized
            return padded

    def color_shift(self, image):
        """颜色偏移."""
        shift = np.random.uniform(-0.1, 0.1, 3)
        shifted = image + shift
        return np.clip(shifted, 0, 1)

    def gaussian_blur(self, image):
        """高斯模糊."""
        kernel_size = random.choice([3, 5])
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred

    def augment(self, image, num_augmentations=1):
        """对图像进行增强."""
        augmented_images = []

        for _ in range(num_augmentations):
            # 随机选择1-3种增强方法
            num_methods = random.randint(1, 3)
            methods = random.sample(self.augmentation_methods, num_methods)

            augmented = image.copy()
            for method in methods:
                augmented = method(augmented)

            augmented_images.append(augmented)

        return augmented_images


def analyze_data_distribution(data_dir):
    """分析数据分布."""
    persons_dir = Path(data_dir) / "persons"

    distribution = {}
    for status_dir in persons_dir.iterdir():
        if status_dir.is_dir():
            count = len(list(status_dir.glob("*.jpg")))
            distribution[status_dir.name] = count

    return distribution


def generate_augmented_data(data_dir, output_dir, target_samples=None):
    """生成增强数据."""
    print("开始生成增强数据...")

    # 分析当前数据分布
    distribution = analyze_data_distribution(data_dir)
    print(f"当前数据分布: {distribution}")

    # 设置目标样本数
    if target_samples is None:
        target_samples = {
            "fully_equipped": 1000,  # 减少主导类别
            "vest_only": 800,
            "helmet_only": 600,  # 大幅增加
            "no_equipment": 500,  # 大幅增加
        }

    print(f"目标样本数: {target_samples}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 创建增强数据目录
    augmented_dir = output_dir / "augmented_data"
    augmented_dir.mkdir(exist_ok=True)

    # 为每个类别创建目录
    for status in target_samples.keys():
        (augmented_dir / status).mkdir(exist_ok=True)

    # 初始化增强器
    augmenter = DataAugmenter()

    # 生成增强数据
    generation_stats = {}

    for status, target_count in target_samples.items():
        print(f"\n处理 {status}...")

        # 获取原始数据
        status_dir = Path(data_dir) / "persons" / status
        if not status_dir.exists():
            print(f"跳过 {status}，目录不存在")
            continue

        original_images = list(status_dir.glob("*.jpg"))
        current_count = len(original_images)

        print(f"原始样本数: {current_count}")
        print(f"目标样本数: {target_count}")

        if current_count >= target_count:
            print(f"{status} 样本已足够，跳过增强")
            # 复制原始数据
            for i, img_path in enumerate(original_images[:target_count]):
                output_path = augmented_dir / status / f"original_{i}.jpg"
                image = cv2.imread(str(img_path))
                cv2.imwrite(str(output_path), image)

            generation_stats[status] = {"original": target_count, "augmented": 0, "total": target_count}
            continue

        # 需要生成的样本数
        needed_samples = target_count - current_count

        # 复制所有原始样本
        for i, img_path in enumerate(original_images):
            output_path = augmented_dir / status / f"original_{i}.jpg"
            image = cv2.imread(str(img_path))
            cv2.imwrite(str(output_path), image)

        # 生成增强样本
        print(f"需要生成 {needed_samples} 个增强样本")

        generated_count = 0
        while generated_count < needed_samples:
            # 随机选择一个原始图像
            source_img_path = random.choice(original_images)

            # 读取图像
            image = cv2.imread(str(source_img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0

            # 生成增强样本
            augmented_images = augmenter.augment(image, num_augmentations=1)

            for aug_img in augmented_images:
                if generated_count >= needed_samples:
                    break

                # 保存增强样本
                aug_img_uint8 = (aug_img * 255).astype(np.uint8)
                aug_img_bgr = cv2.cvtColor(aug_img_uint8, cv2.COLOR_RGB2BGR)

                output_path = augmented_dir / status / f"augmented_{generated_count}.jpg"
                cv2.imwrite(str(output_path), aug_img_bgr)

                generated_count += 1

        generation_stats[status] = {
            "original": current_count,
            "augmented": generated_count,
            "total": current_count + generated_count,
        }

        print(f"完成 {status}: 原始 {current_count} + 增强 {generated_count} = 总计 {current_count + generated_count}")

    # 保存生成统计
    stats_file = output_dir / "augmentation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(
            {
                "original_distribution": distribution,
                "target_samples": target_samples,
                "generation_stats": generation_stats,
                "total_generated": sum(stats["total"] for stats in generation_stats.values()),
            },
            f,
            indent=2,
        )

    print("\n数据增强完成!")
    print(f"增强数据保存在: {augmented_dir}")
    print(f"统计信息保存在: {stats_file}")

    return generation_stats


def create_balanced_dataset(original_data_dir, augmented_data_dir, output_dir):
    """创建平衡的数据集."""
    print("创建平衡的训练数据集...")

    output_dir = Path(output_dir)
    balanced_dir = output_dir / "balanced_dataset"
    balanced_dir.mkdir(exist_ok=True)

    # 创建目录结构
    (balanced_dir / "images").mkdir(exist_ok=True)
    (balanced_dir / "labels").mkdir(exist_ok=True)

    # 复制原始训练数据
    original_images_dir = Path(original_data_dir) / "train" / "images"
    original_labels_dir = Path(original_data_dir) / "train" / "labels"

    print("复制原始训练数据...")
    image_files = list(original_images_dir.glob("*.jpg"))

    for img_file in image_files:
        # 复制图像
        output_img_path = balanced_dir / "images" / img_file.name
        cv2.imwrite(str(output_img_path), cv2.imread(str(img_file)))

        # 复制标签
        label_file = original_labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            output_label_path = balanced_dir / "labels" / (img_file.stem + ".txt")
            with open(label_file) as f:
                content = f.read()
            with open(output_label_path, "w") as f:
                f.write(content)

    print(f"复制了 {len(image_files)} 个原始训练样本")

    # 创建数据集配置文件
    config = {
        "path": str(balanced_dir.absolute()),
        "train": "images",
        "val": str(Path(original_data_dir) / "valid" / "images"),
        "test": str(Path(original_data_dir) / "test" / "images"),
        "nc": 3,
        "names": ["safety_vest", "helmet", "person"],
    }

    with open(balanced_dir / "data.yaml", "w") as f:
        import yaml

        yaml.dump(config, f)

    print(f"平衡数据集创建完成: {balanced_dir}")
    return balanced_dir


def main():
    parser = argparse.ArgumentParser(description="数据增强生成器")
    parser.add_argument("--data", default="vae_training_data", help="原始数据目录")
    parser.add_argument("--original", default="data/railroad-worker-detection", help="原始数据集目录")
    parser.add_argument("--output", default="augmented_dataset", help="输出目录")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"错误: 数据目录不存在 {args.data}")
        return

    # 生成增强数据
    generation_stats = generate_augmented_data(args.data, args.output)

    # 创建平衡的数据集
    balanced_dataset_dir = create_balanced_dataset(args.original, args.output, args.output)

    # 打印总结
    print("\n" + "=" * 60)
    print("数据增强总结")
    print("=" * 60)

    for status, stats in generation_stats.items():
        print(f"{status}:")
        print(f"  原始: {stats['original']}")
        print(f"  增强: {stats['augmented']}")
        print(f"  总计: {stats['total']}")

    total_samples = sum(stats["total"] for stats in generation_stats.values())
    print(f"\n总样本数: {total_samples}")
    print(f"平衡数据集: {balanced_dataset_dir}")


if __name__ == "__main__":
    main()
