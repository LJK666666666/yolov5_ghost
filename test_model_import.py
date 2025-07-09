#!/usr/bin/env python3
"""
测试模型导入和创建

Author: Augment Agent (Claude Sonnet 4 by Anthropic)
Created: 2025-07-05
Description: 测试各种YAML配置文件的模型创建
"""

import sys
import traceback
from pathlib import Path

# 添加项目路径
sys.path.append('.')

try:
    from models.yolo import Model
    print("✅ 成功导入Model类")
except ImportError as e:
    print(f"❌ 导入Model类失败: {e}")
    sys.exit(1)


def test_model_creation(yaml_path, model_name):
    """测试模型创建"""
    print(f"\n🧪 测试 {model_name}")
    print("=" * 50)
    
    if not Path(yaml_path).exists():
        print(f"⚠️  文件不存在: {yaml_path}")
        return False
    
    try:
        print(f"📁 配置文件: {yaml_path}")
        
        # 尝试创建模型
        model = Model(yaml_path, ch=3, nc=3)
        
        # 获取模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建成功")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 测试前向传播
        import torch
        try:
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                y = model(x)
            print(f"✅ 前向传播成功")
            print(f"   输入: {x.shape}")
            print(f"   输出数量: {len(y)}")
            for i, output in enumerate(y):
                print(f"   输出{i}: {output.shape}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        print(f"错误详情:")
        traceback.print_exc()
        return False


def test_class_imports():
    """测试类导入"""
    print("🔍 测试类导入")
    print("=" * 50)
    
    classes_to_test = [
        'SEBlock',
        'C3HybridMoE', 
        'HybridMoELayer',
        'HybridMoEBottleneck',
        'C3MoE',
        'MoELayer'
    ]
    
    for class_name in classes_to_test:
        try:
            # 尝试从common模块导入
            exec(f"from models.common import {class_name}")
            print(f"✅ {class_name} 导入成功")
        except ImportError as e:
            print(f"❌ {class_name} 导入失败: {e}")
        except Exception as e:
            print(f"⚠️  {class_name} 导入异常: {e}")


def main():
    """主函数"""
    print("🧪 模型导入和创建测试")
    print("=" * 80)
    
    # 测试类导入
    test_class_imports()
    
    # 测试模型创建
    models_to_test = [
        ('models/yolov5s.yaml', 'YOLOv5s (标准)'),
        ('models/yolov5s-se.yaml', 'YOLOv5s-SE (SE注意力)'),
        ('models/yolov5s-hybrid-moe.yaml', 'YOLOv5s-Hybrid-MoE (混合MoE)'),
        ('models/yolov5s-sparse-moe.yaml', 'YOLOv5s-Sparse-MoE (稀疏MoE)'),
        ('models/yolov5s-moe-lite.yaml', 'YOLOv5s-MoE-Lite (轻量级MoE)')
    ]
    
    success_count = 0
    total_count = 0
    
    for yaml_path, model_name in models_to_test:
        total_count += 1
        if test_model_creation(yaml_path, model_name):
            success_count += 1
    
    # 总结
    print(f"\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    
    print(f"✅ 成功: {success_count}/{total_count} 个模型")
    print(f"❌ 失败: {total_count - success_count}/{total_count} 个模型")
    
    if success_count == total_count:
        print("\n🎉 所有模型都可以正常创建！")
        print("现在可以安全地进行训练了。")
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个模型创建失败")
        print("请检查相关配置文件和类定义。")
    
    print(f"\n🚀 推荐的训练命令:")
    print("# 使用SE注意力模型")
    print("python train.py --cfg models/yolov5s-se.yaml --data your_data.yaml")
    print("\n# 使用混合MoE模型")  
    print("python train.py --cfg models/yolov5s-hybrid-moe.yaml --data your_data.yaml")


if __name__ == "__main__":
    main()
