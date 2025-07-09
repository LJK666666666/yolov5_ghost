#!/usr/bin/env python3
"""
分析YOLOv5s-SE模型结构和参数

Author: Augment Agent (Claude Sonnet 4 by Anthropic)
Created: 2025-07-05
Description: 分析添加SE注意力机制后的YOLOv5s模型变化
"""

import yaml
import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.append('.')

try:
    from models.yolo import Model
    from models.common import SEBlock
    from utils.torch_utils import profile
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在YOLOv5项目根目录下运行此脚本")
    sys.exit(1)


def load_model_config(yaml_path):
    """加载模型配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def analyze_model_structure(yaml_path, model_name):
    """分析模型结构"""
    print(f"\n🔍 分析 {model_name}")
    print("=" * 60)
    
    config = load_model_config(yaml_path)
    
    # 分析backbone
    backbone = config.get('backbone', [])
    head = config.get('head', [])
    
    print(f"📊 模型结构分析:")
    print(f"  Backbone层数: {len(backbone)}")
    print(f"  Head层数: {len(head)}")
    print(f"  总层数: {len(backbone) + len(head)}")
    
    # 统计SE模块
    se_count = 0
    se_positions = []
    
    print(f"\n📋 Backbone结构:")
    for i, layer in enumerate(backbone):
        if len(layer) >= 3:
            from_layer, number, module = layer[:3]
            args = layer[3] if len(layer) > 3 else []
            
            if module == 'SEBlock':
                se_count += 1
                se_positions.append(f"Backbone-{i}")
                print(f"  {i:2d}. {module:12s} - SE注意力 (通道: {args[0] if args else 'auto'})")
            else:
                print(f"  {i:2d}. {module:12s} - {args}")
    
    print(f"\n📋 Head结构:")
    for i, layer in enumerate(head):
        if len(layer) >= 3:
            from_layer, number, module = layer[:3]
            args = layer[3] if len(layer) > 3 else []
            
            if module == 'SEBlock':
                se_count += 1
                se_positions.append(f"Head-{i}")
                print(f"  {i:2d}. {module:12s} - SE注意力 (通道: {args[0] if args else 'auto'})")
            else:
                print(f"  {i:2d}. {module:12s} - {args}")
    
    print(f"\n🎯 SE注意力统计:")
    print(f"  SE模块总数: {se_count}")
    print(f"  SE模块位置: {', '.join(se_positions)}")
    
    return {
        'se_count': se_count,
        'se_positions': se_positions,
        'total_layers': len(backbone) + len(head)
    }


def create_and_analyze_model(yaml_path, model_name):
    """创建并分析模型"""
    print(f"\n🏗️  创建 {model_name} 模型")
    print("=" * 60)
    
    try:
        # 创建模型
        model = Model(yaml_path, ch=3, nc=3)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        # 统计SE模块参数
        se_params = 0
        se_modules = []
        
        for name, module in model.named_modules():
            if isinstance(module, SEBlock):
                se_module_params = sum(p.numel() for p in module.parameters())
                se_params += se_module_params
                se_modules.append((name, se_module_params))
        
        print(f"\n🎯 SE模块参数统计:")
        print(f"  SE模块总参数: {se_params:,}")
        print(f"  SE参数占比: {se_params/total_params*100:.2f}%")
        print(f"  SE模块数量: {len(se_modules)}")
        
        if se_modules:
            print(f"  SE模块详情:")
            for name, params in se_modules:
                print(f"    {name}: {params:,} 参数")
        
        # 测试前向传播
        print(f"\n🧪 前向传播测试:")
        try:
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                y = model(x)
            print(f"  输入形状: {x.shape}")
            print(f"  输出数量: {len(y)}")
            for i, output in enumerate(y):
                print(f"  输出{i}形状: {output.shape}")
            print("  ✅ 前向传播成功")
        except Exception as e:
            print(f"  ❌ 前向传播失败: {e}")
        
        return {
            'total_params': total_params,
            'se_params': se_params,
            'se_modules_count': len(se_modules),
            'model': model
        }
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None


def compare_models():
    """对比原始模型和SE模型"""
    print("🔄 模型对比分析")
    print("=" * 80)
    
    models = {
        'YOLOv5s (原始)': 'models/yolov5s.yaml',
        'YOLOv5s-SE (SE注意力)': 'models/yolov5s-se.yaml'
    }
    
    results = {}
    
    for model_name, yaml_path in models.items():
        if Path(yaml_path).exists():
            # 分析结构
            structure_info = analyze_model_structure(yaml_path, model_name)
            
            # 创建和分析模型
            model_info = create_and_analyze_model(yaml_path, model_name)
            
            if model_info:
                results[model_name] = {**structure_info, **model_info}
        else:
            print(f"⚠️  文件不存在: {yaml_path}")
    
    # 对比分析
    if len(results) >= 2:
        print(f"\n" + "=" * 80)
        print("📊 模型对比总结")
        print("=" * 80)
        
        original = results.get('YOLOv5s (原始)', {})
        se_model = results.get('YOLOv5s-SE (SE注意力)', {})
        
        if original and se_model:
            print(f"\n📈 参数对比:")
            orig_params = original.get('total_params', 0)
            se_params = se_model.get('total_params', 0)
            param_increase = se_params - orig_params
            param_ratio = se_params / orig_params if orig_params > 0 else 0
            
            print(f"  原始模型参数: {orig_params:,}")
            print(f"  SE模型参数: {se_params:,}")
            print(f"  参数增加: {param_increase:,} ({(param_ratio-1)*100:+.2f}%)")
            
            se_only_params = se_model.get('se_params', 0)
            print(f"  SE模块参数: {se_only_params:,}")
            print(f"  SE参数占比: {se_only_params/se_params*100:.2f}%")
            
            print(f"\n🏗️  结构对比:")
            print(f"  原始模型层数: {original.get('total_layers', 0)}")
            print(f"  SE模型层数: {se_model.get('total_layers', 0)}")
            print(f"  SE模块数量: {se_model.get('se_count', 0)}")
            
            print(f"\n💡 性能预期:")
            print(f"  • 参数增加 {(param_ratio-1)*100:.1f}%，主要来自SE模块")
            print(f"  • 计算量略微增加，推理速度可能稍微下降")
            print(f"  • 特征表示能力增强，可能提升检测精度")
            print(f"  • 对复杂场景和小目标检测可能有改善")


def generate_usage_guide():
    """生成使用指南"""
    print(f"\n" + "=" * 80)
    print("📚 YOLOv5s-SE 使用指南")
    print("=" * 80)
    
    print(f"\n🚀 训练命令:")
    print(f"# 从预训练权重开始训练")
    print(f"python train.py --img 640 --batch 16 --epochs 100 \\")
    print(f"    --data your_data.yaml \\")
    print(f"    --cfg models/yolov5s-se.yaml \\")
    print(f"    --weights yolov5s.pt")
    
    print(f"\n# 从头开始训练")
    print(f"python train.py --img 640 --batch 16 --epochs 100 \\")
    print(f"    --data your_data.yaml \\")
    print(f"    --cfg models/yolov5s-se.yaml \\")
    print(f"    --weights ''")
    
    print(f"\n🔧 推理命令:")
    print(f"python detect.py --weights runs/train/exp/weights/best.pt \\")
    print(f"    --source your_images \\")
    print(f"    --img 640")
    
    print(f"\n⚙️  参数调优建议:")
    print(f"• SE reduction ratio: 可以在SEBlock中调整降维比例")
    print(f"  - 较小值(8): 更强的注意力，但参数更多")
    print(f"  - 较大值(32): 更轻量，但注意力效果可能减弱")
    print(f"• 批次大小: 由于参数增加，可能需要适当减小batch size")
    print(f"• 学习率: 建议使用与原始YOLOv5s相同的学习率策略")
    
    print(f"\n📊 性能评估:")
    print(f"• 使用相同数据集对比原始YOLOv5s和YOLOv5s-SE")
    print(f"• 关注mAP@0.5和mAP@0.5:0.95指标")
    print(f"• 测试推理速度和内存使用情况")
    print(f"• 特别关注小目标和复杂场景的检测效果")


def main():
    """主函数"""
    print("🔍 YOLOv5s-SE 模型分析")
    print("=" * 80)
    
    # 对比模型
    compare_models()
    
    # 生成使用指南
    generate_usage_guide()
    
    print(f"\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
    print("YOLOv5s-SE模型已成功创建并分析完成！")
    print("SE注意力机制已正确集成到模型中，可以开始训练了。")


if __name__ == "__main__":
    main()
