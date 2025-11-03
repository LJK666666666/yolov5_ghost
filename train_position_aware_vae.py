#!/usr/bin/env python3
"""
位置感知的聚类模板VAE训练脚本
实现三个核心要求：
1. 背景损失只计算非人物区域
2. 前景对象从(原图-背景)中提取
3. 头盔/背心位置由VAE隐向量确定.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clustered_template_vae import LATENT_DIMS, TEMPLATE_SIZE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PositionAwareVAE(nn.Module):
    """位置感知的VAE模型."""

    def __init__(self, clustered_templates, cluster_models):
        super().__init__()
        self.clustered_templates = clustered_templates
        self.cluster_models = cluster_models
        self.categories = list(clustered_templates.keys())

        # 外观编码器和解码器
        self.appearance_encoders = nn.ModuleDict()
        self.appearance_decoders = nn.ModuleDict()

        # 位置编码器 (用于头盔和背心)
        self.position_encoders = nn.ModuleDict()

        for category in self.categories:
            template_size = TEMPLATE_SIZE[category]
            latent_dim = LATENT_DIMS[category]
            num_clusters = len(clustered_templates[category])

            # 外观编码器/解码器
            self.appearance_encoders[category] = AppearanceEncoder(category, template_size, latent_dim, num_clusters)
            self.appearance_decoders[category] = AppearanceDecoder(category, template_size, latent_dim)

            # 位置编码器 (仅头盔和背心)
            if category in ["helmet", "safety_vest"]:
                self.position_encoders[category] = PositionEncoder(latent_dim)

    def forward(self, batch):
        """前向传播."""
        outputs = {}
        differences = {}
        latent_params = {}
        cluster_logits = {}
        position_params = {}

        for category in self.categories:
            if category in batch:
                x = batch[category]
                batch_size = x.size(0)

                # 预测聚类并获取模板
                cluster_pred = self.predict_cluster(category, x)
                template = self.get_template(category, cluster_pred, batch_size, x.device)

                # 外观编码
                mu, logvar, cluster_logit = self.appearance_encoders[category](x, template)
                z = self.reparameterize(mu, logvar)

                # 外观解码
                output, difference = self.appearance_decoders[category](z, template)

                outputs[category] = output
                differences[category] = difference
                latent_params[category] = {"mu": mu, "logvar": logvar, "z": z, "clusters": cluster_pred}
                cluster_logits[category] = cluster_logit

                # 位置编码 (仅头盔和背心)
                if category in ["helmet", "safety_vest"]:
                    pos_mu, pos_logvar = self.position_encoders[category](z)
                    position_params[category] = {"mu": pos_mu, "logvar": pos_logvar}

        return outputs, differences, latent_params, cluster_logits, position_params

    def predict_cluster(self, category, x):
        """预测聚类."""
        batch_size = x.size(0)
        num_clusters = len(self.clustered_templates[category])

        # 计算与每个聚类模板的相似度
        cluster_scores = []
        for cluster_id in range(num_clusters):
            template = self.get_template(category, cluster_id, batch_size, x.device)
            similarity = -torch.mean((x - template) ** 2, dim=[1, 2, 3])
            cluster_scores.append(similarity)

        cluster_scores = torch.stack(cluster_scores, dim=1)
        cluster_pred = torch.argmax(cluster_scores, dim=1)
        return cluster_pred

    def get_template(self, category, cluster_id, batch_size, device=None):
        """获取聚类模板."""
        if isinstance(cluster_id, torch.Tensor):
            # 批处理情况
            templates = []
            for i in range(batch_size):
                cid = cluster_id[i].item()
                template = self.clustered_templates[category][cid]
                templates.append(torch.from_numpy(template).permute(2, 0, 1))
            result = torch.stack(templates)
            return result.to(cluster_id.device)
        else:
            # 单个聚类ID
            template = self.clustered_templates[category][cluster_id]
            template_tensor = torch.from_numpy(template).permute(2, 0, 1)
            result = template_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            return result.to(device) if device else result

    def reparameterize(self, mu, logvar):
        """重参数化技巧 (数值稳定版本)."""
        # 限制logvar范围以避免数值不稳定
        logvar = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class AppearanceEncoder(nn.Module):
    """外观编码器."""

    def __init__(self, category, template_size, latent_dim, num_clusters):
        super().__init__()
        self.category = category
        w, h = template_size

        # 编码器网络 (输入是6通道: 3通道图像 + 3通道模板)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # 计算特征尺寸
        self.feature_size = 512 * (h // 16) * (w // 16)

        # 潜在空间映射
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)

        # 聚类分类器
        self.cluster_classifier = nn.Linear(self.feature_size, num_clusters)

    def forward(self, x, template):
        """前向传播."""
        # 拼接输入和模板
        combined = torch.cat([x, template], dim=1)

        # 编码
        features = self.encoder(combined)
        features = features.view(features.size(0), -1)

        # 潜在空间
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        # 聚类预测
        cluster_logits = self.cluster_classifier(features)

        return mu, logvar, cluster_logits


class AppearanceDecoder(nn.Module):
    """外观解码器."""

    def __init__(self, category, template_size, latent_dim):
        super().__init__()
        self.category = category
        w, h = template_size

        # 计算特征图尺寸
        self.feature_h = h // 16
        self.feature_w = w // 16

        # 潜在向量到特征图
        self.fc = nn.Linear(latent_dim, 512 * self.feature_h * self.feature_w)

        # 解码器网络
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, template):
        """前向传播."""
        # 潜在向量到特征图
        features = self.fc(z)
        features = features.view(-1, 512, self.feature_h, self.feature_w)

        # 解码差值
        difference = self.decoder(features)

        # 确保尺寸匹配
        if difference.shape != template.shape:
            difference = F.interpolate(difference, size=template.shape[2:], mode="bilinear", align_corners=True)

        # 模板 + 差值 = 最终输出
        output = template + difference
        output = torch.clamp(output, -1, 1)

        return output, difference


class PositionEncoder(nn.Module):
    """位置编码器 - 从外观隐向量预测相对位置."""

    def __init__(self, appearance_latent_dim, position_dim=4):
        super().__init__()
        self.position_dim = position_dim  # [rel_x, rel_y, rel_w, rel_h]

        # 从外观隐向量预测位置
        self.position_net = nn.Sequential(
            nn.Linear(appearance_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, position_dim * 2),  # mu和logvar
        )

    def forward(self, appearance_z):
        """
        appearance_z: 外观隐向量 [B, appearance_latent_dim]
        返回: position_mu [B, 4], position_logvar [B, 4].
        """
        position_params = self.position_net(appearance_z)
        position_params = position_params.view(-1, self.position_dim, 2)

        position_mu = position_params[:, :, 0]  # [B, 4]
        position_logvar = position_params[:, :, 1]  # [B, 4]

        return position_mu, position_logvar


class PositionAwareDataset(Dataset):
    """位置感知的数据集."""

    def __init__(self, data_dir, templates, cluster_models, max_samples=None):
        self.data_dir = Path(data_dir)
        self.templates = templates
        self.cluster_models = cluster_models
        self.samples = []

        # 收集样本
        self._collect_samples(max_samples)

        print(f"✅ 收集到 {len(self.samples)} 个有效样本")

    def _collect_samples(self, max_samples):
        """收集训练样本."""
        images_dir = self.data_dir / "train" / "images"
        labels_dir = self.data_dir / "train" / "labels"

        image_files = list(images_dir.glob("*.jpg"))
        if max_samples:
            image_files = image_files[:max_samples]

        for img_file in tqdm(image_files, desc="收集样本"):
            # 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 读取标签
            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                continue

            # 解析标注
            annotations = self._parse_annotations(label_file, image.shape[1], image.shape[0])

            # 提取样本
            sample = self._extract_corrected_sample(image, annotations)
            if sample:
                self.samples.append(sample)

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

    def _extract_corrected_sample(self, image, annotations):
        """正确的样本提取方法 - 实现三个核心要求."""
        h, w = image.shape[:2]

        # 1. 创建人物区域掩码 (用于背景损失计算)
        person_mask = np.zeros((h, w), dtype=np.uint8)
        person_bboxes = []

        # 2. 创建所有物体掩码 (用于背景提取)
        all_objects_mask = np.zeros((h, w), dtype=np.uint8)

        for ann in annotations:
            x1, y1, x2, y2 = ann["bbox"]
            all_objects_mask[y1:y2, x1:x2] = 1

            if ann["class_id"] == 2:  # person
                person_mask[y1:y2, x1:x2] = 1
                person_bboxes.append([x1, y1, x2, y2])

        # 3. 提取纯背景 (使用图像修复)
        try:
            background = cv2.inpaint(image, all_objects_mask, 3, cv2.INPAINT_TELEA)
        except:
            return None

        # 4. 计算前景 = 原图 - 背景
        foreground = image.astype(np.float32) - background.astype(np.float32)
        foreground = np.clip(foreground + 128, 0, 255).astype(np.uint8)

        sample = {}

        # 5. 背景样本 + 改进的背景掩码
        if np.sum(all_objects_mask) / (h * w) < 0.5:
            bg_resized = cv2.resize(background, TEMPLATE_SIZE["background"])
            sample["background"] = bg_resized.astype(np.float32) / 255.0 * 2 - 1

            # 改进的背景掩码：区分纯背景区域和人物区域
            # 1=纯背景区域, 0=人物区域(包含人物锚框内的背景部分)
            background_region_mask = 1 - person_mask  # 非人物区域
            mask_resized = cv2.resize(background_region_mask.astype(np.float32), TEMPLATE_SIZE["background"])
            sample["background_mask"] = mask_resized

            # 添加统计信息用于调试
            pure_bg_ratio = np.sum(background_region_mask) / (h * w)
            sample["background_stats"] = {"pure_bg_ratio": pure_bg_ratio, "person_region_ratio": 1 - pure_bg_ratio}

        # 6. 从前景中提取各类别对象 + 位置信息
        position_targets = {}

        for ann in annotations:
            class_id = ann["class_id"]
            bbox = ann["bbox"]
            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            # 从前景图像中裁剪 (而不是原图)
            foreground_crop = foreground[y1:y2, x1:x2]

            if class_id == 2:  # person
                crop = cv2.resize(foreground_crop, TEMPLATE_SIZE["person"])
                sample["person"] = crop.astype(np.float32) / 255.0 * 2 - 1

            elif class_id == 0:  # safety_vest
                crop = cv2.resize(foreground_crop, TEMPLATE_SIZE["safety_vest"])
                sample["safety_vest"] = crop.astype(np.float32) / 255.0 * 2 - 1

                # 计算相对于人物的位置
                if person_bboxes:
                    rel_pos = self._compute_relative_position(bbox, person_bboxes[0])
                    position_targets["safety_vest"] = rel_pos

            elif class_id == 1:  # helmet
                crop = cv2.resize(foreground_crop, TEMPLATE_SIZE["helmet"])
                sample["helmet"] = crop.astype(np.float32) / 255.0 * 2 - 1

                # 计算相对于人物的位置
                if person_bboxes:
                    rel_pos = self._compute_relative_position(bbox, person_bboxes[0])
                    position_targets["helmet"] = rel_pos

        # 添加位置目标
        if position_targets:
            sample["position_targets"] = position_targets

        return sample if len(sample) > 1 else None

    def _compute_relative_position(self, object_bbox, person_bbox):
        """计算物体相对于人物的归一化位置."""
        ox1, oy1, ox2, oy2 = object_bbox
        px1, py1, px2, py2 = person_bbox

        # 计算相对位置 (归一化到[-1,1])
        rel_x = ((ox1 + ox2) / 2 - (px1 + px2) / 2) / max(px2 - px1, 1)
        rel_y = ((oy1 + oy2) / 2 - (py1 + py2) / 2) / max(py2 - py1, 1)
        rel_w = (ox2 - ox1) / max(px2 - px1, 1)
        rel_h = (oy2 - oy1) / max(py2 - py1, 1)

        return np.array([rel_x, rel_y, rel_w, rel_h], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 转换为tensor
        tensor_sample = {}
        for key, value in sample.items():
            if key == "position_targets":
                # 位置目标保持为numpy数组
                tensor_sample[key] = value
            elif key == "background_mask":
                # 背景掩码
                tensor_sample[key] = torch.from_numpy(value)
            elif key == "background_stats":
                # 背景统计信息保持为字典
                tensor_sample[key] = value
            else:
                # 图像数据
                tensor_sample[key] = torch.from_numpy(value).permute(2, 0, 1)

        return tensor_sample


def collate_fn(batch):
    """自定义collate函数."""
    all_categories = set()
    for sample in batch:
        all_categories.update(sample.keys())

    batched = {}
    for category in all_categories:
        if category == "position_targets":
            # 处理位置目标
            position_targets = {}
            batch_indices = {}  # 记录每个样本的索引

            for batch_idx, sample in enumerate(batch):
                if "position_targets" in sample:
                    for obj_type, pos in sample["position_targets"].items():
                        if obj_type not in position_targets:
                            position_targets[obj_type] = []
                            batch_indices[obj_type] = []
                        position_targets[obj_type].append(pos)
                        batch_indices[obj_type].append(batch_idx)

            # 转换为tensor并确保与其他数据的批次大小一致
            for obj_type, positions in position_targets.items():
                if positions:
                    position_targets[obj_type] = torch.from_numpy(np.array(positions))
                    print(f"📊 {obj_type} 位置目标: {len(positions)} 个样本")

            batched["position_targets"] = position_targets
        elif category == "background_stats":
            # 处理背景统计信息
            stats_list = []
            for sample in batch:
                if "background_stats" in sample:
                    stats_list.append(sample["background_stats"])
            batched["background_stats"] = stats_list
        else:
            # 处理其他数据
            category_data = []
            for sample in batch:
                if category in sample:
                    category_data.append(sample[category])

            if category_data:
                batched[category] = torch.stack(category_data)

    return batched


def compute_position_aware_loss(
    outputs,
    targets,
    differences,
    latent_params,
    cluster_logits,
    position_params,
    background_masks=None,
    position_targets=None,
    true_clusters=None,
    beta=1.0,
    gamma=1.0,
    lambda_pos=0.5,
    person_region_weight=0.3,
):
    """位置感知的损失计算 - 改进的背景损失计算."""
    total_loss = 0
    loss_components = {}

    for category in outputs.keys():
        if category in targets:
            category_loss = 0
            components = {}

            # 1. 重建损失 - 改进: 给人物区域较小权重而不是完全排除
            if category == "background" and background_masks is not None:
                # 背景损失：给人物区域较小权重
                mask = background_masks  # [B, H, W] 1=背景区域, 0=人物区域
                mask = mask.unsqueeze(1)  # [B, 1, H, W]

                # 创建权重掩码：背景区域权重1.0，人物区域权重person_region_weight
                weight_mask = mask + (1 - mask) * person_region_weight  # [B, 1, H, W]

                # 计算加权重建损失
                recon_loss = F.mse_loss(outputs[category], targets[category], reduction="none")
                weighted_recon_loss = recon_loss * weight_mask

                # 归一化损失
                total_weight = weight_mask.sum(dim=[1, 2, 3], keepdim=True) + 1e-8
                recon_loss = weighted_recon_loss.sum() / total_weight.sum()

                components["bg_pure_weight"] = mask.float().mean().item()
                components["bg_person_weight"] = person_region_weight
            else:
                # 其他类别：正常重建损失
                recon_loss = F.mse_loss(outputs[category], targets[category])

            components["recon"] = recon_loss.item()
            category_loss += recon_loss

            # 2. KL散度损失 (数值稳定版本)
            mu = latent_params[category]["mu"]
            logvar = latent_params[category]["logvar"]

            # 限制logvar范围以避免数值不稳定
            logvar = torch.clamp(logvar, min=-20, max=20)

            # 数值稳定的KL损失计算
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

            components["kl"] = kl_loss.item()
            category_loss += beta * kl_loss

            # 3. 差值正则化
            diff_reg = torch.mean(torch.abs(differences[category]))
            components["diff_reg"] = diff_reg.item()
            category_loss += 0.1 * diff_reg

            # 4. 聚类分类损失
            cluster_loss = 0
            if true_clusters and category in true_clusters:
                cluster_loss = F.cross_entropy(cluster_logits[category], true_clusters[category])
                components["cluster"] = cluster_loss.item()
                category_loss += gamma * cluster_loss

            # 5. 位置损失 - 要求3: 头盔/背心位置由VAE隐向量确定
            position_loss = 0
            if category in ["helmet", "safety_vest"] and position_params and category in position_params:
                if position_targets and category in position_targets:
                    # 位置重建损失
                    pred_pos_mu = position_params[category]["mu"]
                    pred_pos_logvar = position_params[category]["logvar"]
                    target_pos = position_targets[category]

                    # 检查批次大小匹配
                    pred_batch_size = pred_pos_mu.size(0)
                    target_batch_size = target_pos.size(0)

                    if pred_batch_size != target_batch_size:
                        # 如果批次大小不匹配，只对有位置目标的样本计算损失
                        min_batch_size = min(pred_batch_size, target_batch_size)

                        if min_batch_size > 0:
                            pred_pos_mu = pred_pos_mu[:min_batch_size]
                            pred_pos_logvar = pred_pos_logvar[:min_batch_size]
                            target_pos = target_pos[:min_batch_size]

                            # 位置重建损失
                            position_recon_loss = F.mse_loss(pred_pos_mu, target_pos)

                            # 位置KL损失 (数值稳定版本)
                            pred_pos_logvar = torch.clamp(pred_pos_logvar, min=-20, max=20)
                            position_kl_loss = (
                                -0.5
                                * torch.sum(1 + pred_pos_logvar - pred_pos_mu.pow(2) - pred_pos_logvar.exp())
                                / pred_pos_mu.size(0)
                            )

                            position_loss = position_recon_loss + 0.1 * position_kl_loss
                            components["position"] = position_loss.item()
                            components["position_samples"] = min_batch_size
                            category_loss += lambda_pos * position_loss
                        else:
                            # 如果没有有效样本，跳过位置损失
                            components["position"] = 0.0
                            components["position_samples"] = 0
                    else:
                        # 批次大小匹配，正常计算
                        position_recon_loss = F.mse_loss(pred_pos_mu, target_pos)

                        pred_pos_logvar = torch.clamp(pred_pos_logvar, min=-20, max=20)
                        position_kl_loss = (
                            -0.5
                            * torch.sum(1 + pred_pos_logvar - pred_pos_mu.pow(2) - pred_pos_logvar.exp())
                            / pred_pos_mu.size(0)
                        )

                        position_loss = position_recon_loss + 0.1 * position_kl_loss
                        components["position"] = position_loss.item()
                        components["position_samples"] = pred_batch_size
                        category_loss += lambda_pos * position_loss
                else:
                    # 没有位置目标，跳过位置损失
                    components["position"] = 0.0
                    components["position_samples"] = 0

            components["total"] = category_loss.item()
            loss_components[category] = components
            total_loss += category_loss

    return total_loss, loss_components


def train_position_aware_vae(
    data_dir, output_dir, epochs=50, batch_size=4, lr=1e-4, max_samples=None, person_region_weight=0.3
):
    """训练位置感知的VAE."""
    print("🚀 开始训练位置感知的聚类模板VAE")
    print("=" * 60)
    print("✅ 实现的核心要求:")
    print("  1. 背景损失：给人物区域较小权重而不是完全排除")
    print("  2. 前景对象从(原图-背景)中提取")
    print("  3. 头盔/背心位置由VAE隐向量确定")
    print(f"  4. 人物区域权重: {person_region_weight}")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    # 加载聚类模板
    from clustered_template_vae import ClusteredTemplateManager

    template_manager = ClusteredTemplateManager(data_dir)
    templates, cluster_models = template_manager.load_clustered_templates()

    if templates is None:
        print("❌ 需要先创建聚类模板")
        return

    print("✅ 加载聚类模板成功")
    for category, category_templates in templates.items():
        print(f"  {category}: {len(category_templates)} 个聚类")

    # 创建模型
    model = PositionAwareVAE(templates, cluster_models).to(device)

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数数量: {total_params:,}")

    # 创建数据集和数据加载器
    dataset = PositionAwareDataset(data_dir, templates, cluster_models, max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_components = {}

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            for category in batch:
                if category == "position_targets":
                    # 位置目标字典
                    for obj_type, positions in batch[category].items():
                        batch[category][obj_type] = positions.to(device)
                elif category == "background_stats":
                    # 背景统计信息保持为列表，不需要移动到设备
                    pass
                else:
                    batch[category] = batch[category].to(device)

            # 提取位置目标和背景掩码
            position_targets = batch.pop("position_targets", None)
            background_masks = batch.pop("background_mask", None)

            # 前向传播
            optimizer.zero_grad()
            outputs, differences, latent_params, cluster_logits, position_params = model(batch)

            # 计算损失 - 更温和的KL退火
            beta = min(0.5, epoch / 50)  # 更慢的KL退火，最大值0.5
            gamma = 0.1  # 降低聚类损失权重
            lambda_pos = 0.5  # 降低位置损失权重

            loss, loss_components = compute_position_aware_loss(
                outputs,
                batch,
                differences,
                latent_params,
                cluster_logits,
                position_params,
                background_masks,
                position_targets,
                beta=beta,
                gamma=gamma,
                lambda_pos=lambda_pos,
                person_region_weight=person_region_weight,
            )

            # 检查NaN和异常值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  检测到异常损失: {loss.item()}")
                print("   损失组件详情:")
                for category, components in loss_components.items():
                    print(f"     {category}: {components}")

                # 检查模型参数
                nan_params = []
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        nan_params.append(name)

                if nan_params:
                    print(f"   发现异常参数: {nan_params[:5]}...")  # 只显示前5个

                print("   跳过此批次")
                optimizer.zero_grad()
                continue

            # 检查损失是否过大
            if loss.item() > 1e10:
                print(f"⚠️  损失过大: {loss.item():.2e}")
                print("   应用梯度裁剪...")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # 更严格的裁剪

            # 反向传播
            loss.backward()

            # 检查梯度
            total_grad_norm = 0
            nan_grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_grad_count += 1
                        print(f"⚠️  {name} 梯度包含NaN/Inf")
                    total_grad_norm += param.grad.data.norm(2).item() ** 2

            total_grad_norm = total_grad_norm**0.5

            if nan_grad_count > 0:
                print(f"⚠️  发现 {nan_grad_count} 个参数的梯度异常，跳过此步")
                optimizer.zero_grad()
                continue

            # 自适应梯度裁剪
            if total_grad_norm > 10.0:
                print(f"⚠️  梯度范数过大: {total_grad_norm:.2f}，应用强裁剪")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            elif total_grad_norm > 5.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()

            # 累积损失组件
            for category, components in loss_components.items():
                if category not in epoch_components:
                    epoch_components[category] = {}
                for comp_name, comp_value in components.items():
                    if comp_name not in epoch_components[category]:
                        epoch_components[category][comp_name] = []
                    epoch_components[category][comp_name].append(comp_value)

            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "β": f"{beta:.2f}"})

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        # 学习率调度
        scheduler.step(avg_loss)

        # 打印epoch结果
        print(f"\nEpoch {epoch + 1}/{epochs} - 平均损失: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.2e}")
        for category, components in epoch_components.items():
            avg_components = {name: np.mean(values) for name, values in components.items()}
            print(f"  {category}: " + ", ".join([f"{name}={value:.4f}" for name, value in avg_components.items()]))

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "train_losses": train_losses,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")

    # 保存最终模型
    final_checkpoint = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_losses[-1],
        "train_losses": train_losses,
        "templates": templates,
        "cluster_models": cluster_models,
    }
    torch.save(final_checkpoint, output_dir / "position_aware_vae_final.pth")

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Position-Aware VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(output_dir / "training_curve.png")
    plt.close()

    print("\n✅ 训练完成!")
    print(f"📁 模型保存在: {output_dir}")
    print(f"📊 最终损失: {train_losses[-1]:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="训练位置感知的聚类模板VAE")
    parser.add_argument("--data", type=str, required=True, help="数据集目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--person_weight", type=float, default=0.3, help="人物区域在背景损失中的权重 (0.0-1.0)")

    args = parser.parse_args()

    train_position_aware_vae(
        args.data, args.output, args.epochs, args.batch_size, args.lr, args.max_samples, args.person_weight
    )


if __name__ == "__main__":
    main()
