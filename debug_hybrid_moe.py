#!/usr/bin/env python3
"""
调试HybridMoE模型创建问题.

Author: Augment Agent (Claude Sonnet 4 by Anthropic)
Created: 2025-07-05
"""

import sys
import traceback

import torch

sys.path.append(".")


def test_individual_classes():
    """测试各个类的单独创建."""
    print("🔍 测试各个类的单独创建")
    print("=" * 50)

    try:
        from models.common import C3HybridMoE, Expert, HybridMoEBottleneck, HybridMoELayer, SharedExpert, SparseGating

        print("✅ 所有类导入成功")
    except ImportError as e:
        print(f"❌ 类导入失败: {e}")
        return

    # 测试SharedExpert
    try:
        SharedExpert(c1=64, c2=32)
        print("✅ SharedExpert 创建成功")
    except Exception as e:
        print(f"❌ SharedExpert 创建失败: {e}")
        traceback.print_exc()

    # 测试Expert
    try:
        Expert(c1=64, c2=32)
        print("✅ Expert 创建成功")
    except Exception as e:
        print(f"❌ Expert 创建失败: {e}")
        traceback.print_exc()

    # 测试SparseGating
    try:
        SparseGating(c1=64, num_experts=4, top_k=2)
        print("✅ SparseGating 创建成功")
    except Exception as e:
        print(f"❌ SparseGating 创建失败: {e}")
        traceback.print_exc()

    # 测试HybridMoELayer
    try:
        HybridMoELayer(c1=64, c2=64, num_experts=4, top_k=2, shared_ratio=0.25)
        print("✅ HybridMoELayer 创建成功")
    except Exception as e:
        print(f"❌ HybridMoELayer 创建失败: {e}")
        traceback.print_exc()

    # 测试HybridMoEBottleneck
    try:
        HybridMoEBottleneck(c1=64, c2=64, num_experts=4, top_k=2, shared_ratio=0.25)
        print("✅ HybridMoEBottleneck 创建成功")
    except Exception as e:
        print(f"❌ HybridMoEBottleneck 创建失败: {e}")
        traceback.print_exc()

    # 测试C3HybridMoE
    try:
        C3HybridMoE(c1=64, c2=64, n=1, num_experts=4, top_k=2, shared_ratio=0.25)
        print("✅ C3HybridMoE 创建成功")
    except Exception as e:
        print(f"❌ C3HybridMoE 创建失败: {e}")
        traceback.print_exc()


def test_forward_pass():
    """测试前向传播."""
    print("\n🧪 测试前向传播")
    print("=" * 50)

    try:
        from models.common import C3HybridMoE

        # 创建模型
        model = C3HybridMoE(c1=64, c2=64, n=1, num_experts=4, top_k=2, shared_ratio=0.25)

        # 创建输入
        x = torch.randn(1, 64, 32, 32)

        # 前向传播
        with torch.no_grad():
            y = model(x)

        print("✅ 前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {y.shape}")

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        traceback.print_exc()


def test_yaml_parsing():
    """测试YAML解析."""
    print("\n📄 测试YAML解析")
    print("=" * 50)

    try:
        from models.yolo import parse_model

        # 简化的YAML配置
        simple_config = {
            "backbone": [[-1, 1, "Conv", [64, 6, 2, 2]], [-1, 1, "C3HybridMoE", [64, True, 1, 0.5, 4, 2, 0.25]]],
            "head": [[[-1], 1, "Detect", [3, [[10, 13, 16, 30, 33, 23]]]]],
        }

        print("🔧 尝试解析简化配置...")
        model, save = parse_model(simple_config, ch=[3])
        print("✅ YAML解析成功")

    except Exception as e:
        print(f"❌ YAML解析失败: {e}")
        traceback.print_exc()


def main():
    """主函数."""
    print("🐛 HybridMoE 调试")
    print("=" * 80)

    test_individual_classes()
    test_forward_pass()
    test_yaml_parsing()

    print("\n" + "=" * 80)
    print("📋 调试完成")
    print("如果所有测试都通过，说明HybridMoE实现正常。")
    print("如果有失败，请检查错误信息进行修复。")


if __name__ == "__main__":
    main()
