#!/usr/bin/env python3
"""
高级场景生成器 - 基于训练好的位置感知VAE
支持多种生成模式和参数控制
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_position_aware_vae import PositionAwareVAE
from vae.clustered_template_vae import TEMPLATE_SIZE

class AdvancedSceneGenerator:
    """高级场景生成器"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.templates = checkpoint['templates']
        self.cluster_models = checkpoint['cluster_models']
        
        self.model = PositionAwareVAE(self.templates, self.cluster_models).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 模型加载成功: {model_path}")
        print(f"🔧 使用设备: {self.device}")
        
        # 打印可用的聚类信息
        print(f"📊 可用聚类:")
        for category, category_templates in self.templates.items():
            print(f"  {category}: {len(category_templates)} 个聚类")
    
    def generate_controlled_scene(self, bg_cluster=None, person_cluster=None, 
                                helmet_cluster=None, vest_cluster=None,
                                person_position=None, scene_size=(360, 640)):
        """生成可控制的场景"""
        with torch.no_grad():
            # 1. 生成背景
            background = self._generate_background(bg_cluster)
            
            # 2. 生成人物
            person_data = self._generate_person(person_cluster, person_position, scene_size)
            
            # 3. 生成头盔和背心
            helmet_data = self._generate_helmet(helmet_cluster, person_data['position'])
            vest_data = self._generate_vest(vest_cluster, person_data['position'])
            
            # 4. 合成场景
            final_scene = self._compose_scene_advanced(
                background, person_data, helmet_data, vest_data, scene_size
            )
            
            return {
                'scene': final_scene,
                'components': {
                    'background': background,
                    'person': person_data,
                    'helmet': helmet_data,
                    'vest': vest_data
                },
                'metadata': {
                    'bg_cluster': background['cluster_id'],
                    'person_cluster': person_data['cluster_id'],
                    'helmet_cluster': helmet_data['cluster_id'],
                    'vest_cluster': vest_data['cluster_id'],
                    'scene_size': scene_size
                }
            }
    
    def _generate_background(self, cluster_id=None):
        """生成背景"""
        bg_clusters = list(self.templates['background'].keys())
        if cluster_id is None:
            cluster_id = np.random.choice(bg_clusters)
        else:
            cluster_id = min(cluster_id, len(bg_clusters) - 1)
        
        # 获取模板
        template = self.templates['background'][cluster_id]
        template_tensor = torch.from_numpy(template).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 采样隐向量
        latent_dim = 256
        z = torch.randn(1, latent_dim, device=self.device)
        
        # 解码
        output, _ = self.model.appearance_decoders['background'](z, template_tensor)
        
        return {
            'image': output[0],
            'cluster_id': cluster_id,
            'latent': z[0]
        }
    
    def _generate_person(self, cluster_id=None, position=None, scene_size=(360, 640)):
        """生成人物"""
        person_clusters = list(self.templates['person'].keys())
        if cluster_id is None:
            cluster_id = np.random.choice(person_clusters)
        else:
            cluster_id = min(cluster_id, len(person_clusters) - 1)
        
        # 获取模板
        template = self.templates['person'][cluster_id]
        template_tensor = torch.from_numpy(template).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 采样隐向量
        latent_dim = 128
        z = torch.randn(1, latent_dim, device=self.device)
        
        # 解码外观
        output, _ = self.model.appearance_decoders['person'](z, template_tensor)
        
        # 生成或使用指定位置
        if position is None:
            scene_x = np.random.uniform(0.2, 0.8)
            scene_y = np.random.uniform(0.5, 0.8)
            scene_w = np.random.uniform(0.15, 0.25)
            scene_h = np.random.uniform(0.3, 0.5)
            position = [scene_x, scene_y, scene_w, scene_h]
        
        return {
            'image': output[0],
            'cluster_id': cluster_id,
            'latent': z[0],
            'position': position
        }
    
    def _generate_helmet(self, cluster_id=None, person_position=None):
        """生成头盔"""
        helmet_clusters = list(self.templates['helmet'].keys())
        if cluster_id is None:
            cluster_id = np.random.choice(helmet_clusters)
        else:
            cluster_id = min(cluster_id, len(helmet_clusters) - 1)
        
        # 获取模板
        template = self.templates['helmet'][cluster_id]
        template_tensor = torch.from_numpy(template).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 采样隐向量
        latent_dim = 32
        z = torch.randn(1, latent_dim, device=self.device)
        
        # 解码外观
        output, _ = self.model.appearance_decoders['helmet'](z, template_tensor)
        
        # 使用位置编码器预测相对位置
        if 'helmet' in self.model.position_encoders:
            pos_mu, pos_logvar = self.model.position_encoders['helmet'](z)
            # 采样位置
            pos_std = torch.exp(0.5 * pos_logvar)
            pos_eps = torch.randn_like(pos_std)
            relative_pos = pos_mu + pos_eps * pos_std
            relative_pos = relative_pos[0].cpu().numpy()
        else:
            # 默认头盔位置
            relative_pos = np.array([0.0, -0.4, 0.3, 0.2])
        
        # 转换为绝对位置
        if person_position:
            person_x, person_y, person_w, person_h = person_position
            helmet_x = person_x + relative_pos[0] * person_w
            helmet_y = person_y + relative_pos[1] * person_h
            helmet_w = abs(relative_pos[2]) * person_w
            helmet_h = abs(relative_pos[3]) * person_h
            absolute_pos = [helmet_x, helmet_y, helmet_w, helmet_h]
        else:
            absolute_pos = [0.5, 0.3, 0.1, 0.1]
        
        return {
            'image': output[0],
            'cluster_id': cluster_id,
            'latent': z[0],
            'relative_position': relative_pos,
            'absolute_position': absolute_pos
        }
    
    def _generate_vest(self, cluster_id=None, person_position=None):
        """生成背心"""
        vest_clusters = list(self.templates['safety_vest'].keys())
        if cluster_id is None:
            cluster_id = np.random.choice(vest_clusters)
        else:
            cluster_id = min(cluster_id, len(vest_clusters) - 1)
        
        # 获取模板
        template = self.templates['safety_vest'][cluster_id]
        template_tensor = torch.from_numpy(template).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 采样隐向量
        latent_dim = 64
        z = torch.randn(1, latent_dim, device=self.device)
        
        # 解码外观
        output, _ = self.model.appearance_decoders['safety_vest'](z, template_tensor)
        
        # 使用位置编码器预测相对位置
        if 'safety_vest' in self.model.position_encoders:
            pos_mu, pos_logvar = self.model.position_encoders['safety_vest'](z)
            # 采样位置
            pos_std = torch.exp(0.5 * pos_logvar)
            pos_eps = torch.randn_like(pos_std)
            relative_pos = pos_mu + pos_eps * pos_std
            relative_pos = relative_pos[0].cpu().numpy()
        else:
            # 默认背心位置
            relative_pos = np.array([0.0, 0.1, 0.6, 0.4])
        
        # 转换为绝对位置
        if person_position:
            person_x, person_y, person_w, person_h = person_position
            vest_x = person_x + relative_pos[0] * person_w
            vest_y = person_y + relative_pos[1] * person_h
            vest_w = abs(relative_pos[2]) * person_w
            vest_h = abs(relative_pos[3]) * person_h
            absolute_pos = [vest_x, vest_y, vest_w, vest_h]
        else:
            absolute_pos = [0.5, 0.6, 0.15, 0.2]
        
        return {
            'image': output[0],
            'cluster_id': cluster_id,
            'latent': z[0],
            'relative_position': relative_pos,
            'absolute_position': absolute_pos
        }
    
    def _compose_scene_advanced(self, background, person_data, helmet_data, vest_data, scene_size):
        """高级场景合成"""
        scene_h, scene_w = scene_size
        
        # 创建场景画布
        scene = torch.zeros(3, scene_h, scene_w, device=self.device)
        
        # 1. 放置背景
        bg_resized = F.interpolate(
            background['image'].unsqueeze(0), 
            size=(scene_h, scene_w), 
            mode='bilinear', 
            align_corners=True
        )[0]
        scene = bg_resized
        
        # 2. 放置人物
        person_pos = person_data['position']
        person_x = int(person_pos[0] * scene_w - person_pos[2] * scene_w / 2)
        person_y = int(person_pos[1] * scene_h - person_pos[3] * scene_h / 2)
        person_w = int(person_pos[2] * scene_w)
        person_h = int(person_pos[3] * scene_h)
        
        # 确保边界
        person_x = max(0, min(person_x, scene_w - person_w))
        person_y = max(0, min(person_y, scene_h - person_h))
        person_w = min(person_w, scene_w - person_x)
        person_h = min(person_h, scene_h - person_y)
        
        if person_w > 0 and person_h > 0:
            person_resized = F.interpolate(
                person_data['image'].unsqueeze(0),
                size=(person_h, person_w),
                mode='bilinear',
                align_corners=True
            )[0]
            scene[:, person_y:person_y+person_h, person_x:person_x+person_w] = person_resized
        
        # 3. 放置头盔
        helmet_pos = helmet_data['absolute_position']
        helmet_x = int(helmet_pos[0] * scene_w - helmet_pos[2] * scene_w / 2)
        helmet_y = int(helmet_pos[1] * scene_h - helmet_pos[3] * scene_h / 2)
        helmet_w = int(abs(helmet_pos[2]) * scene_w)
        helmet_h = int(abs(helmet_pos[3]) * scene_h)
        
        # 确保边界
        helmet_x = max(0, min(helmet_x, scene_w - helmet_w))
        helmet_y = max(0, min(helmet_y, scene_h - helmet_h))
        helmet_w = min(helmet_w, scene_w - helmet_x)
        helmet_h = min(helmet_h, scene_h - helmet_y)
        
        if helmet_w > 0 and helmet_h > 0:
            helmet_resized = F.interpolate(
                helmet_data['image'].unsqueeze(0),
                size=(helmet_h, helmet_w),
                mode='bilinear',
                align_corners=True
            )[0]
            scene[:, helmet_y:helmet_y+helmet_h, helmet_x:helmet_x+helmet_w] = helmet_resized
        
        # 4. 放置背心
        vest_pos = vest_data['absolute_position']
        vest_x = int(vest_pos[0] * scene_w - vest_pos[2] * scene_w / 2)
        vest_y = int(vest_pos[1] * scene_h - vest_pos[3] * scene_h / 2)
        vest_w = int(abs(vest_pos[2]) * scene_w)
        vest_h = int(abs(vest_pos[3]) * scene_h)
        
        # 确保边界
        vest_x = max(0, min(vest_x, scene_w - vest_w))
        vest_y = max(0, min(vest_y, scene_h - vest_h))
        vest_w = min(vest_w, scene_w - vest_x)
        vest_h = min(vest_h, scene_h - vest_y)
        
        if vest_w > 0 and vest_h > 0:
            vest_resized = F.interpolate(
                vest_data['image'].unsqueeze(0),
                size=(vest_h, vest_w),
                mode='bilinear',
                align_corners=True
            )[0]
            scene[:, vest_y:vest_y+vest_h, vest_x:vest_x+vest_w] = vest_resized
        
        return scene

def tensor_to_numpy(tensor):
    """将tensor转换为numpy图像"""
    img = (tensor + 1) / 2
    img = torch.clamp(img, 0, 1)
    img = img.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img

def generate_diverse_scenes(model_path, output_dir, num_samples=20):
    """生成多样化的场景"""
    print("🎨 生成多样化的安全设备检测场景...")
    
    generator = AdvancedSceneGenerator(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成随机场景
    print("📸 生成随机场景...")
    for i in tqdm(range(num_samples), desc="随机场景"):
        scene_data = generator.generate_controlled_scene()
        
        # 保存场景
        scene_img = tensor_to_numpy(scene_data['scene'])
        cv2.imwrite(
            str(output_dir / f'random_scene_{i:03d}.jpg'),
            cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR)
        )
        
        # 保存元数据
        metadata = scene_data['metadata'].copy()
        metadata['components'] = {
            'person_position': [float(x) for x in scene_data['components']['person']['position']],
            'helmet_relative_pos': [float(x) for x in scene_data['components']['helmet']['relative_position'].tolist()],
            'vest_relative_pos': [float(x) for x in scene_data['components']['vest']['relative_position'].tolist()],
        }

        # 确保所有值都是JSON可序列化的
        for key, value in metadata.items():
            if hasattr(value, 'item'):  # numpy scalar
                metadata[key] = value.item()
            elif isinstance(value, (np.int64, np.int32)):
                metadata[key] = int(value)
        
        with open(output_dir / f'random_scene_{i:03d}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # 生成控制场景 - 不同聚类组合
    print("🎯 生成控制场景...")
    bg_clusters = list(generator.templates['background'].keys())
    person_clusters = list(generator.templates['person'].keys())
    
    scene_idx = 0
    for bg_id in bg_clusters[:3]:  # 前3个背景聚类
        for person_id in person_clusters[:3]:  # 前3个人物聚类
            scene_data = generator.generate_controlled_scene(
                bg_cluster=bg_id, 
                person_cluster=person_id
            )
            
            scene_img = tensor_to_numpy(scene_data['scene'])
            cv2.imwrite(
                str(output_dir / f'controlled_scene_{scene_idx:03d}_bg{bg_id}_person{person_id}.jpg'),
                cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR)
            )
            
            scene_idx += 1
    
    print(f"✅ 生成完成! 保存在: {output_dir}")
    print(f"📊 总共生成了 {num_samples + scene_idx} 个场景")

def main():
    parser = argparse.ArgumentParser(description='高级场景生成器')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--num_samples', type=int, default=20, help='随机场景数量')
    
    args = parser.parse_args()
    
    generate_diverse_scenes(args.model, args.output, args.num_samples)

if __name__ == "__main__":
    main()
