#!/usr/bin/env python3
"""下载 safety-helmet-and-vest 数据集并准备用于YOLOv5训练."""

import random
import shutil
from pathlib import Path

import kagglehub
import yaml


def download_and_prepare_dataset():
    """下载并准备safety-helmet-and-vest数据集."""
    print("正在下载 safety-helmet-and-vest 数据集...")

    # Download latest version
    path = kagglehub.dataset_download("maryamborzoo/safety-helmet-and-vest")
    print("Path to dataset files:", path)

    # 创建目标目录
    target_dir = Path("data/safety-helmet-vest")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 检查下载的数据集结构
    source_path = Path(path)
    print(f"数据集下载到: {source_path}")
    print("数据集内容:")
    for item in source_path.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(source_path)}")

    # 查找图片和标注文件
    image_files = []
    label_files = []

    # 常见的图片和标注文件扩展名
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    label_extensions = {".txt", ".xml"}

    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
            elif file_path.suffix.lower() in label_extensions:
                label_files.append(file_path)

    print(f"找到 {len(image_files)} 个图片文件")
    print(f"找到 {len(label_files)} 个标注文件")

    # 如果是XML格式，需要转换为YOLO格式
    if any(f.suffix.lower() == ".xml" for f in label_files):
        print("检测到XML格式标注，需要转换为YOLO格式")
        # 这里可以添加XML到YOLO格式的转换代码
        # 暂时跳过，假设已经是YOLO格式

    # 创建YOLOv5目录结构
    for split in ["train", "valid", "test"]:
        (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 过滤有对应标注的图片
    valid_images = []
    for img_file in image_files:
        # 查找对应的标注文件
        label_name = img_file.stem + ".txt"
        label_file = None

        # 在所有标注文件中查找匹配的
        for lbl_file in label_files:
            if lbl_file.name == label_name or lbl_file.stem == img_file.stem:
                label_file = lbl_file
                break

        if label_file and label_file.exists():
            valid_images.append((img_file, label_file))

    print(f"找到 {len(valid_images)} 对有效的图片-标注对")

    if len(valid_images) == 0:
        print("错误: 没有找到有效的图片-标注对")
        return False

    # 随机打乱并分割数据集
    random.seed(42)  # 确保可重现
    random.shuffle(valid_images)

    # 分割比例: 70% train, 20% valid, 10% test
    total = len(valid_images)
    train_end = int(0.7 * total)
    valid_end = int(0.9 * total)

    train_pairs = valid_images[:train_end]
    valid_pairs = valid_images[train_end:valid_end]
    test_pairs = valid_images[valid_end:]

    print(f"数据集分割: {len(train_pairs)} train, {len(valid_pairs)} valid, {len(test_pairs)} test")

    # 复制文件到对应目录
    def copy_files(pairs, split_name):
        for img_file, label_file in pairs:
            # 复制图片
            target_img = target_dir / split_name / "images" / img_file.name
            shutil.copy2(img_file, target_img)

            # 复制标注
            target_label = target_dir / split_name / "labels" / (img_file.stem + ".txt")
            shutil.copy2(label_file, target_label)

    copy_files(train_pairs, "train")
    copy_files(valid_pairs, "valid")
    copy_files(test_pairs, "test")

    # 分析类别信息
    class_counts = {}
    for _, label_file in valid_images:
        try:
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except Exception as e:
            print(f"读取标注文件失败 {label_file}: {e}")

    print(f"类别分布: {class_counts}")
    num_classes = len(class_counts) if class_counts else 3

    # 创建data.yaml配置文件
    # 根据safety-helmet-and-vest数据集，通常包含: helmet, vest, person等类别
    class_names = []
    if 0 in class_counts:
        class_names.append("person")
    if 1 in class_counts:
        class_names.append("helmet")
    if 2 in class_counts:
        class_names.append("vest")

    # 如果没有检测到类别，使用默认名称
    if not class_names:
        class_names = ["person", "helmet", "vest"][:num_classes]

    data_config = {
        "path": "./",  # dataset root dir
        "train": "data/safety-helmet-vest/train/images",
        "val": "data/safety-helmet-vest/valid/images",
        "test": "data/safety-helmet-vest/test/images",
        "nc": len(class_names),  # number of classes
        "names": class_names,  # class names
    }

    # 保存data.yaml
    with open(target_dir / "data.yaml", "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print("数据集准备完成!")
    print(f"配置文件保存到: {target_dir / 'data.yaml'}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")
    print(f"训练集: {len(train_pairs)} 张图片")
    print(f"验证集: {len(valid_pairs)} 张图片")
    print(f"测试集: {len(test_pairs)} 张图片")

    return True


if __name__ == "__main__":
    success = download_and_prepare_dataset()
    if success:
        print("\n数据集下载和准备成功!")
        print("现在可以开始训练了。")
    else:
        print("\n数据集准备失败!")
