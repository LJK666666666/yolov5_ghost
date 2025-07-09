#!/usr/bin/env python3
"""
增强VAE的生成脚本
适配EnhancedPersonCentricVAE模型
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse

from enhanced_person_vae import (
    EnhancedPersonCentricVAE, LATENT_DIMS, ORIGINAL_SIZE, 
    composite_scene_enhanced
)

class EnhancedGenerator:
    """增强VAE生成器"""
    
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"增强VAE生成器已加载，使用设备: {self.device}")
    
    def _load_model(self, model_path):
        """加载训练好的增强模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 使用增强版模型
        model = EnhancedPersonCentricVAE(LATENT_DIMS).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"模型加载成功，训练轮数: {checkpoint.get('epoch', 'unknown')}")
        print(f"验证损失: {checkpoint.get('val_loss', 'unknown'):.4f}")
        if 'detection_loss' in checkpoint:
            print(f"检测损失: {checkpoint['detection_loss']:.4f}")
        
        return model
    
    def sample_latent_factors(self, batch_size=1, controls=None):
        """采样隐向量"""
        latent_factors = {}
        
        for factor_name, dim in LATENT_DIMS.items():
            if controls and factor_name in controls:
                # 使用指定的控制参数
                if isinstance(controls[factor_name], torch.Tensor):
                    latent_factors[factor_name] = controls[factor_name].to(self.device)
                else:
                    # 从标准正态分布采样
                    z = torch.randn(batch_size, dim, device=self.device)
                    latent_factors[factor_name] = z
            else:
                # 从标准正态分布采样
                latent_factors[factor_name] = torch.randn(batch_size, dim, device=self.device)
        
        return latent_factors
    
    def generate_controlled(self, controls=None, num_samples=1):
        """可控生成"""
        with torch.no_grad():
            # 采样隐向量
            latent_factors = self.sample_latent_factors(num_samples, controls)
            
            # 解码背景
            background = self.model.background_decoder(latent_factors['background'])
            
            # 解码人物
            person_decoded = self.model.person_decoder(latent_factors)
            
            decoded = {
                'background': background,
                'person': person_decoded
            }
            
            # 平滑合成完整场景
            final_images = composite_scene_enhanced(
                decoded['background'], 
                decoded['person'], 
                target_size=ORIGINAL_SIZE,
                feather=25  # 增强羽化效果
            )
            
            return final_images, decoded
    
    def generate_random_samples(self, num_samples=1):
        """生成随机样本"""
        return self.generate_controlled({}, num_samples)
    
    def generate_style_variations(self, num_variations=5, fixed_factors=None):
        """生成风格变化"""
        variations = []
        
        with torch.no_grad():
            for _ in range(num_variations):
                controls = {}
                
                # 固定某些因子，变化其他因子
                if fixed_factors:
                    for factor_name, factor_value in fixed_factors.items():
                        if factor_name in LATENT_DIMS:
                            controls[factor_name] = factor_value
                
                final_image, _ = self.generate_controlled(controls, 1)
                variations.append(final_image)
        
        return torch.cat(variations, dim=0)
    
    def interpolate_factors(self, factor_name, start_z=None, end_z=None, steps=10):
        """在指定因子上进行插值"""
        if start_z is None:
            start_z = torch.randn(1, LATENT_DIMS[factor_name], device=self.device)
        if end_z is None:
            end_z = torch.randn(1, LATENT_DIMS[factor_name], device=self.device)
        
        interpolated_images = []
        
        with torch.no_grad():
            for i in range(steps):
                alpha = i / (steps - 1)
                
                # 插值
                interpolated_z = (1 - alpha) * start_z + alpha * end_z
                
                # 生成其他因子
                latent_factors = self.sample_latent_factors(1)
                latent_factors[factor_name] = interpolated_z
                
                # 解码和合成
                background = self.model.background_decoder(latent_factors['background'])
                person_decoded = self.model.person_decoder(latent_factors)
                
                decoded = {
                    'background': background,
                    'person': person_decoded
                }
                
                final_image = composite_scene_enhanced(
                    decoded['background'], 
                    decoded['person'], 
                    target_size=ORIGINAL_SIZE,
                    feather=25
                )
                
                interpolated_images.append(final_image)
        
        return torch.cat(interpolated_images, dim=0)

def tensor_to_numpy(tensor):
    """将tensor转换为numpy图像"""
    # 反标准化
    tensor = (tensor + 1) / 2  # [-1,1] -> [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为numpy
    img_np = tensor.cpu().detach().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # 确保正确的尺寸 (应该是1920x1080)
    print(f"生成图像尺寸: {img_np.shape}")
    
    # BGR格式用于OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    return img_np

def generate_enhanced_samples(model_path, output_dir, num_samples=10):
    """生成增强VAE样本"""
    print("开始生成增强VAE样本...")
    
    generator = EnhancedGenerator(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 随机生成
    print("生成随机样本...")
    random_images, _ = generator.generate_random_samples(num_samples)
    
    for i, img_tensor in enumerate(random_images):
        img_np = tensor_to_numpy(img_tensor)
        cv2.imwrite(str(output_dir / f'enhanced_random_{i:03d}.jpg'), img_np)
    
    # 2. 风格变化 - 固定背景，变化人物
    print("生成风格变化...")
    fixed_background = torch.randn(1, LATENT_DIMS['background'], device=generator.device)
    fixed_factors = {'background': fixed_background}
    
    style_variations = generator.generate_style_variations(8, fixed_factors)
    
    for i, img_tensor in enumerate(style_variations):
        img_np = tensor_to_numpy(img_tensor)
        cv2.imwrite(str(output_dir / f'enhanced_style_{i:03d}.jpg'), img_np)
    
    # 3. 因子插值 - 人物身体变化
    print("生成人物身体插值...")
    body_interpolated = generator.interpolate_factors('person_body', steps=8)
    
    for i, img_tensor in enumerate(body_interpolated):
        img_np = tensor_to_numpy(img_tensor)
        cv2.imwrite(str(output_dir / f'enhanced_body_interp_{i:03d}.jpg'), img_np)
    
    # 4. 因子插值 - 头盔样式变化
    print("生成头盔样式插值...")
    helmet_interpolated = generator.interpolate_factors('helmet_style', steps=8)
    
    for i, img_tensor in enumerate(helmet_interpolated):
        img_np = tensor_to_numpy(img_tensor)
        cv2.imwrite(str(output_dir / f'enhanced_helmet_interp_{i:03d}.jpg'), img_np)
    
    # 5. 因子插值 - 背心样式变化
    print("生成背心样式插值...")
    vest_interpolated = generator.interpolate_factors('vest_style', steps=8)
    
    for i, img_tensor in enumerate(vest_interpolated):
        img_np = tensor_to_numpy(img_tensor)
        cv2.imwrite(str(output_dir / f'enhanced_vest_interp_{i:03d}.jpg'), img_np)
    
    # 6. 因子插值 - 背景变化
    print("生成背景插值...")
    background_interpolated = generator.interpolate_factors('background', steps=8)
    
    for i, img_tensor in enumerate(background_interpolated):
        img_np = tensor_to_numpy(img_tensor)
        cv2.imwrite(str(output_dir / f'enhanced_background_interp_{i:03d}.jpg'), img_np)
    
    print(f"增强VAE样本生成完成! 保存在: {output_dir}")

def create_quality_comparison(model_path, output_dir):
    """创建质量对比"""
    print("创建质量对比...")
    
    generator = EnhancedGenerator(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成不同羽化程度的对比
    print("生成融合效果对比...")
    
    # 固定参数
    latent_factors = generator.sample_latent_factors(1)
    background = generator.model.background_decoder(latent_factors['background'])
    person_decoded = generator.model.person_decoder(latent_factors)
    
    decoded = {
        'background': background,
        'person': person_decoded
    }
    
    # 不同羽化程度
    feather_values = [0, 10, 20, 30, 40]
    
    for feather in feather_values:
        final_image = composite_scene_enhanced(
            decoded['background'], 
            decoded['person'], 
            target_size=ORIGINAL_SIZE,
            feather=feather
        )
        
        img_np = tensor_to_numpy(final_image[0])
        cv2.imwrite(str(output_dir / f'enhanced_feather_{feather:02d}.jpg'), img_np)
    
    print(f"质量对比完成! 保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='增强VAE生成')
    parser.add_argument('--model', default='enhanced_vae_results/enhanced_vae_best.pth', help='模型路径')
    parser.add_argument('--output', default='enhanced_generation_results', help='输出目录')
    parser.add_argument('--mode', choices=['samples', 'comparison'], default='samples', help='生成模式')
    parser.add_argument('--num_samples', type=int, default=10, help='生成样本数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 {args.model}")
        return
    
    if args.mode == 'samples':
        generate_enhanced_samples(args.model, args.output, args.num_samples)
    elif args.mode == 'comparison':
        create_quality_comparison(args.model, args.output)
    
    print("生成完成!")

if __name__ == "__main__":
    main()
