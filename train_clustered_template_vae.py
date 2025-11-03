#!/usr/bin/env python3
"""训练基于聚类模板的VAE."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vae.clustered_template_model import ClusteredTemplateVAE, compute_clustered_template_loss
from vae.clustered_template_vae import TEMPLATE_SIZE, ClusteredTemplateManager


class ClusteredTemplateDataset(Dataset):
    """基于聚类模板的数据集."""

    def __init__(self, data_dir, templates, cluster_models, max_samples=None):
        self.data_dir = Path(data_dir)
        self.templates = templates
        self.cluster_models = cluster_models
        self.samples = []
        self.precomputed_labels_count = 0
        self.computed_labels_count = 0

        # 收集样本
        self._collect_samples(max_samples)

    def _collect_samples(self, max_samples):
        """收集训练样本."""
        images_dir = self.data_dir / "train" / "images"
        labels_dir = self.data_dir / "train" / "labels"

        image_files = list(images_dir.glob("*.jpg"))
        if max_samples:
            image_files = image_files[:max_samples]

        print(f"收集聚类模板样本中... (最大 {max_samples or '无限制'} 个)")

        for img_file in tqdm(image_files):
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
            annotations = self._parse_annotations(label_file, w, h)

            # 提取各类别样本
            sample = self._extract_sample(image, annotations, img_file.name)
            if sample:
                self.samples.append(sample)

        print(f"✅ 收集到 {len(self.samples)} 个有效样本")
        print("📊 聚类标签统计:")
        print(f"  预创建标签: {self.precomputed_labels_count} 个")
        print(f"  运行时计算: {self.computed_labels_count} 个")
        if self.precomputed_labels_count > 0:
            efficiency = (
                self.precomputed_labels_count / (self.precomputed_labels_count + self.computed_labels_count) * 100
            )
            print(f"  效率提升: {efficiency:.1f}% 使用预创建标签")

    def _parse_annotations(self, label_file, img_width, img_height):
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

    def _extract_sample(self, image, annotations, img_filename=None):
        """提取样本并分配聚类标签."""
        sample = {}
        cluster_labels = {}
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
                background = background.astype(np.float32) / 255.0 * 2 - 1  # [-1,1]
                sample["background"] = background
            except:
                pass

        # 提取物体
        for ann in annotations:
            class_id = ann["class_id"]
            bbox = ann["bbox"]
            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if class_id == 2:  # person
                crop = cv2.resize(crop, TEMPLATE_SIZE["person"])
                crop = crop.astype(np.float32) / 255.0 * 2 - 1
                sample["person"] = crop

            elif class_id == 0:  # safety_vest
                crop = cv2.resize(crop, TEMPLATE_SIZE["safety_vest"])
                crop = crop.astype(np.float32) / 255.0 * 2 - 1
                sample["safety_vest"] = crop

            elif class_id == 1:  # helmet
                crop = cv2.resize(crop, TEMPLATE_SIZE["helmet"])
                crop = crop.astype(np.float32) / 255.0 * 2 - 1
                sample["helmet"] = crop

        # 加载预创建的聚类标签（如果存在）
        if img_filename and sample:
            cluster_labels = self._load_cluster_labels(img_filename)
            if cluster_labels:
                # 将图像级别的聚类标签转换为样本级别
                sample_cluster_labels = self._convert_image_labels_to_sample_labels(cluster_labels, sample)
                self.precomputed_labels_count += 1
            else:
                # 如果没有预创建的标签，则运行时计算
                sample_cluster_labels = self._compute_cluster_labels(sample)
                self.computed_labels_count += 1
            sample["cluster_labels"] = sample_cluster_labels
            return sample
        return None

    def _load_cluster_labels(self, img_filename):
        """加载预创建的聚类标签."""
        try:
            # 构建聚类标签文件路径
            img_stem = Path(img_filename).stem
            cluster_label_file = self.data_dir / "train" / "cluster_labels" / f"{img_stem}.json"

            if cluster_label_file.exists():
                with open(cluster_label_file) as f:
                    cluster_labels = json.load(f)
                return cluster_labels
            else:
                return None
        except Exception:
            return None

    def _convert_image_labels_to_sample_labels(self, image_cluster_labels, sample):
        """将图像级别的聚类标签转换为样本级别."""
        sample_cluster_labels = {}

        for category, data in sample.items():
            if category != "cluster_labels" and category in image_cluster_labels:
                image_labels = image_cluster_labels[category]

                if isinstance(image_labels, list):
                    # 如果是列表，取第一个（简化处理）
                    # 实际上这里应该根据提取的样本在原图中的位置来匹配
                    sample_cluster_labels[category] = image_labels[0] if image_labels else 0
                else:
                    # 如果是单个值（如背景）
                    sample_cluster_labels[category] = image_labels
            elif category != "cluster_labels":
                # 如果没有预创建标签，运行时计算
                sample_cluster_labels[category] = self._assign_cluster_label(category, data)

        return sample_cluster_labels

    def _compute_cluster_labels(self, sample):
        """运行时计算聚类标签（备用方案）."""
        cluster_labels = {}

        for category, data in sample.items():
            if category != "cluster_labels":
                cluster_labels[category] = self._assign_cluster_label(category, data)

        return cluster_labels

    def _assign_cluster_label(self, category, image_data):
        """为图像分配聚类标签."""
        if category not in self.cluster_models:
            return 0

        # 提取特征 (简化版本，使用颜色直方图)
        image_np = ((image_data + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0,255]

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
        cluster_label = self.cluster_models[category].predict([features])[0]

        return cluster_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 转换为tensor
        tensor_sample = {}
        for category, data in sample.items():
            if category == "cluster_labels":
                tensor_sample[category] = data  # 保持字典格式
            else:
                tensor_sample[category] = torch.from_numpy(data).permute(2, 0, 1)  # HWC -> CHW

        return tensor_sample


def collate_fn(batch):
    """自定义collate函数 - 处理聚类标签."""
    # 收集所有类别
    all_categories = set()
    for sample in batch:
        all_categories.update(sample.keys())

    # 为每个类别创建batch
    batched = {}
    for category in all_categories:
        if category == "cluster_labels":
            # 处理聚类标签
            cluster_labels = {}
            for sample in batch:
                if "cluster_labels" in sample:
                    for cat, label in sample["cluster_labels"].items():
                        if cat not in cluster_labels:
                            cluster_labels[cat] = []
                        cluster_labels[cat].append(label)

            # 转换为tensor
            for cat, labels in cluster_labels.items():
                cluster_labels[cat] = torch.tensor(labels, dtype=torch.long)

            batched["cluster_labels"] = cluster_labels
        else:
            # 处理图像数据
            category_data = []
            for sample in batch:
                if category in sample:
                    category_data.append(sample[category])

            if category_data:
                batched[category] = torch.stack(category_data)

    return batched


def train_clustered_template_vae(data_dir, output_dir, epochs=30, batch_size=4, lr=1e-4, max_samples=300):
    """训练聚类模板VAE."""
    print("🚀 开始训练基于聚类模板的VAE")
    print("=" * 50)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载聚类模板
    template_manager = ClusteredTemplateManager(data_dir)
    templates, cluster_models = template_manager.load_clustered_templates()

    if templates is None:
        print("❌ 需要先创建聚类模板")
        return None

    # 创建模型
    model = ClusteredTemplateVAE(templates, cluster_models).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 数据集
    dataset = ClusteredTemplateDataset(data_dir, templates, cluster_models, max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(dataloader)}")

    # 训练循环
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_components = {}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            for category in batch:
                if category == "cluster_labels":
                    # 处理聚类标签字典
                    for cat, labels in batch[category].items():
                        batch[category][cat] = labels.to(device)
                else:
                    batch[category] = batch[category].to(device)

            # 提取真实聚类标签
            true_clusters = batch.pop("cluster_labels", None)

            # 前向传播
            optimizer.zero_grad()
            outputs, differences, latent_params, cluster_logits = model(batch)

            # 计算损失
            beta = min(1.0, epoch / 10)  # KL退火
            gamma = 0.5  # 聚类损失权重 (增加权重)
            loss, loss_components = compute_clustered_template_loss(
                outputs,
                batch,
                differences,
                latent_params,
                cluster_logits,
                true_clusters=true_clusters,
                beta=beta,
                gamma=gamma,
            )

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print("⚠️  检测到NaN损失，跳过此批次")
                print(f"  批次组成: {list(batch.keys())}")
                for cat, components in loss_components.items():
                    print(f"  {cat}: {components}")
                optimizer.zero_grad()
                continue

            # 反向传播
            loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()

            # 更新组件损失
            for category, components in loss_components.items():
                if category not in epoch_components:
                    epoch_components[category] = {k: [] for k in components.keys()}
                for k, v in components.items():
                    epoch_components[category][k].append(v)

            # 更新进度条
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "β": f"{beta:.3f}"})

        # 记录epoch损失
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")

        # 打印各类别损失
        for category, components in epoch_components.items():
            avg_components = {k: np.mean(v) for k, v in components.items()}
            print(
                f"  {category}: Recon={avg_components['recon']:.4f}, "
                f"KL={avg_components['kl']:.4f}, "
                f"DiffReg={avg_components['diff_reg']:.4f}, "
                f"Cluster={avg_components['cluster']:.4f}"
            )

        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "templates": templates,
                    "cluster_models": cluster_models,
                },
                output_dir / f"clustered_template_vae_epoch_{epoch + 1}.pth",
            )

    # 保存最终模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "templates": templates,
            "cluster_models": cluster_models,
            "train_losses": train_losses,
        },
        output_dir / "clustered_template_vae_final.pth",
    )

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Clustered Template VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(output_dir / "training_loss.png")
    plt.close()

    print(f"✅ 训练完成！模型已保存到: {output_dir}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练基于聚类模板的VAE")
    parser.add_argument("--data", default="data/railroad-worker-detection", help="数据目录")
    parser.add_argument("--output", default="clustered_template_vae_results", help="输出目录")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max_samples", type=int, default=250, help="最大样本数")

    args = parser.parse_args()

    train_clustered_template_vae(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
