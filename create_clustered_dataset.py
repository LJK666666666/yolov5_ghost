#!/usr/bin/env python3
"""
创建带聚类标签的数据集
基于现有的聚类模型为每个样本分配聚类标签.
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from clustered_template_vae import TEMPLATE_SIZE, ClusteredTemplateManager
from tqdm import tqdm


def create_clustered_dataset(source_dir, output_dir):
    """创建带聚类标签的数据集."""
    print("🎨 创建带聚类标签的数据集...")
    print("=" * 50)

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # 创建输出目录结构
    output_dir.mkdir(parents=True, exist_ok=True)

    # 复制原始结构
    for split in ["train", "val"]:
        if (source_dir / split).exists():
            # 复制图像和标签
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "cluster_labels").mkdir(parents=True, exist_ok=True)

            print(f"复制 {split} 数据...")

            # 复制图像
            images_src = source_dir / split / "images"
            images_dst = output_dir / split / "images"
            if images_src.exists():
                for img_file in images_src.glob("*.jpg"):
                    shutil.copy2(img_file, images_dst / img_file.name)

            # 复制标签
            labels_src = source_dir / split / "labels"
            labels_dst = output_dir / split / "labels"
            if labels_src.exists():
                for label_file in labels_src.glob("*.txt"):
                    shutil.copy2(label_file, labels_dst / label_file.name)

    # 加载聚类模型
    template_manager = ClusteredTemplateManager(source_dir)
    templates, cluster_models = template_manager.load_clustered_templates()

    if templates is None or cluster_models is None:
        print("❌ 需要先创建聚类模板和模型")
        return

    print("✅ 加载聚类模型成功")
    for category, category_templates in templates.items():
        print(f"  {category}: {len(category_templates)} 个聚类")

    # 为每个split处理聚类标签
    cluster_stats = defaultdict(lambda: defaultdict(int))
    cluster_samples = defaultdict(lambda: defaultdict(list))

    for split in ["train", "val"]:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue

        print(f"\n🔄 处理 {split} 数据的聚类标签...")

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        cluster_labels_dir = split_dir / "cluster_labels"

        image_files = list(images_dir.glob("*.jpg"))

        for img_file in tqdm(image_files, desc=f"处理{split}"):
            # 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # 读取标签
            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                continue

            # 解析标注
            annotations = parse_annotations(label_file, w, h)

            # 提取各类别样本并分配聚类
            sample_data = extract_and_cluster_sample(image, annotations, cluster_models, templates)

            if sample_data:
                # 保存聚类标签
                cluster_label_file = cluster_labels_dir / (img_file.stem + ".json")
                with open(cluster_label_file, "w") as f:
                    json.dump(sample_data["cluster_labels"], f, indent=2)

                # 统计聚类分布
                for category, cluster_id in sample_data["cluster_labels"].items():
                    if isinstance(cluster_id, list):
                        for cid in cluster_id:
                            cluster_stats[category][cid] += 1
                            cluster_samples[category][cid].append(img_file.name)
                    else:
                        cluster_stats[category][cluster_id] += 1
                        cluster_samples[category][cluster_id].append(img_file.name)

    # 保存聚类统计信息
    save_cluster_analysis(output_dir, cluster_stats, cluster_samples, templates)

    # 创建聚类可视化
    create_cluster_visualization(output_dir, cluster_samples, templates)

    print("\n✅ 聚类标签数据集创建完成!")
    print(f"📁 输出目录: {output_dir}")
    print("📊 聚类统计已保存")


def parse_annotations(label_file, img_width, img_height):
    """解析YOLO标注."""
    annotations = []

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

                x1 = max(0, int(x_center - width / 2))
                y1 = max(0, int(y_center - height / 2))
                x2 = min(img_width, int(x_center + width / 2))
                y2 = min(img_height, int(y_center + height / 2))

                annotations.append({"class_id": class_id, "bbox": [x1, y1, x2, y2]})

    return annotations


def extract_and_cluster_sample(image, annotations, cluster_models, templates):
    """提取样本并分配聚类标签."""
    sample_data = {"cluster_labels": {}}
    h, w = image.shape[:2]

    # 提取背景
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        mask[y1:y2, x1:x2] = 1

    if np.sum(mask) / (h * w) < 0.5:  # 物体占比不超过50%
        try:
            background = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            background = cv2.resize(background, TEMPLATE_SIZE["background"])

            # 分配背景聚类
            bg_cluster = assign_cluster_label("background", background, cluster_models)
            sample_data["cluster_labels"]["background"] = bg_cluster
        except:
            pass

    # 提取物体并分配聚类
    person_clusters = []
    vest_clusters = []
    helmet_clusters = []

    for ann in annotations:
        class_id = ann["class_id"]
        bbox = ann["bbox"]
        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        if class_id == 2:  # person
            crop = cv2.resize(crop, TEMPLATE_SIZE["person"])
            cluster_id = assign_cluster_label("person", crop, cluster_models)
            person_clusters.append(cluster_id)

        elif class_id == 0:  # safety_vest
            crop = cv2.resize(crop, TEMPLATE_SIZE["safety_vest"])
            cluster_id = assign_cluster_label("safety_vest", crop, cluster_models)
            vest_clusters.append(cluster_id)

        elif class_id == 1:  # helmet
            crop = cv2.resize(crop, TEMPLATE_SIZE["helmet"])
            cluster_id = assign_cluster_label("helmet", crop, cluster_models)
            helmet_clusters.append(cluster_id)

    # 保存物体聚类列表
    if person_clusters:
        sample_data["cluster_labels"]["person"] = person_clusters
    if vest_clusters:
        sample_data["cluster_labels"]["safety_vest"] = vest_clusters
    if helmet_clusters:
        sample_data["cluster_labels"]["helmet"] = helmet_clusters

    return sample_data if sample_data["cluster_labels"] else None


def assign_cluster_label(category, image_data, cluster_models):
    """为图像分配聚类标签."""
    if category not in cluster_models:
        return 0

    # 提取特征 (与训练时相同的方法)
    image_np = (image_data * 255).astype(np.uint8)

    # 颜色直方图特征
    hist_r = cv2.calcHist([image_np], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image_np], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([image_np], [2], None, [32], [0, 256])

    # 归一化
    hist_r = hist_r.flatten() / np.sum(hist_r)
    hist_g = hist_g.flatten() / np.sum(hist_g)
    hist_b = hist_b.flatten() / np.sum(hist_b)

    # 平均颜色和标准差
    mean_color = np.mean(image_np.reshape(-1, 3), axis=0) / 255.0
    std_color = np.std(image_np.reshape(-1, 3), axis=0) / 255.0

    # 组合特征
    features = np.concatenate([hist_r, hist_g, hist_b, mean_color, std_color])

    # 使用聚类模型预测
    cluster_label = cluster_models[category].predict([features])[0]

    return int(cluster_label)


def save_cluster_analysis(output_dir, cluster_stats, cluster_samples, templates):
    """保存聚类分析结果."""
    analysis_dir = output_dir / "cluster_analysis"
    analysis_dir.mkdir(exist_ok=True)

    # 保存统计信息
    stats_data = {
        "cluster_distribution": dict(cluster_stats),
        "total_samples": {category: sum(counts.values()) for category, counts in cluster_stats.items()},
        "cluster_info": {
            category: {"num_clusters": len(templates[category]), "cluster_sizes": dict(counts)}
            for category, counts in cluster_stats.items()
        },
    }

    with open(analysis_dir / "cluster_stats.json", "w") as f:
        json.dump(stats_data, f, indent=2)

    # 保存样本列表
    with open(analysis_dir / "cluster_samples.json", "w") as f:
        # 转换为可序列化的格式
        serializable_samples = {}
        for category, clusters in cluster_samples.items():
            serializable_samples[category] = dict(clusters)
        json.dump(serializable_samples, f, indent=2)

    print("📊 聚类分析结果:")
    for category, counts in cluster_stats.items():
        print(f"  {category}:")
        for cluster_id, count in sorted(counts.items()):
            print(f"    聚类 {cluster_id}: {count} 个样本")


def create_cluster_visualization(output_dir, cluster_samples, templates):
    """创建聚类可视化."""
    vis_dir = output_dir / "cluster_visualization"
    vis_dir.mkdir(exist_ok=True)

    print("🎨 创建聚类可视化...")

    for category in templates.keys():
        category_dir = vis_dir / category
        category_dir.mkdir(exist_ok=True)

        if category in cluster_samples:
            for cluster_id, sample_files in cluster_samples[category].items():
                cluster_dir = category_dir / f"cluster_{cluster_id}"
                cluster_dir.mkdir(exist_ok=True)

                # 创建说明文件
                info = {
                    "cluster_id": cluster_id,
                    "num_samples": len(sample_files),
                    "sample_files": sample_files[:20],  # 只保存前20个作为示例
                }

                with open(cluster_dir / "info.json", "w") as f:
                    json.dump(info, f, indent=2)

    print(f"✅ 聚类可视化已创建: {vis_dir}")


if __name__ == "__main__":
    source_dir = "data/railroad-worker-detection"
    output_dir = "data/railroad-worker-detection-clustered"

    if Path(source_dir).exists():
        create_clustered_dataset(source_dir, output_dir)
    else:
        print(f"❌ 源数据集不存在: {source_dir}")
