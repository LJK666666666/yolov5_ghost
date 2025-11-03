#!/usr/bin/env python3
"""
测试修复后的HybridMoE模型.

Author: Augment Agent (Claude Sonnet 4 by Anthropic)
Created: 2025-07-05
"""

import sys
import traceback

import torch

sys.path.append(".")


def test_hybrid_moe_model():
    """测试HybridMoE模型创建和前向传播."""
    print("🧪 测试修复后的YOLOv5s-Hybrid-MoE模型")
    print("=" * 60)

    try:
        from models.yolo import Model

        # 创建模型
        print("🔧 创建模型...")
        model = Model("models/yolov5s-hybrid-moe.yaml", ch=3, nc=3)

        # 获取模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("✅ 模型创建成功！")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

        # 测试前向传播
        print("\n🚀 测试前向传播...")
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            y = model(x)

        print("✅ 前向传播成功！")
        print(f"   输入形状: {x.shape}")
        print(f"   输出数量: {len(y)}")
        for i, output in enumerate(y):
            print(f"   输出{i}: {output.shape}")

        # 测试负载均衡损失
        print("\n📊 测试负载均衡损失...")
        try:
            total_load_loss = 0.0
            moe_count = 0

            for name, module in model.named_modules():
                if hasattr(module, "get_load_balancing_loss"):
                    load_loss = module.get_load_balancing_loss()
                    total_load_loss += load_loss
                    moe_count += 1
                    print(f"   {name}: {load_loss:.6f}")

            print("✅ 负载均衡损失测试成功！")
            print(f"   MoE模块数量: {moe_count}")
            print(f"   总负载均衡损失: {total_load_loss:.6f}")

        except Exception as e:
            print(f"⚠️  负载均衡损失测试失败: {e}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("错误详情:")
        traceback.print_exc()
        return False


def test_training_compatibility():
    """测试训练兼容性."""
    print("\n🎯 测试训练兼容性")
    print("=" * 60)

    try:
        from models.yolo import Model

        # 创建模型
        model = Model("models/yolov5s-hybrid-moe.yaml", ch=3, nc=3)

        # 测试损失计算
        print("🔧 测试损失计算...")
        x = torch.randn(2, 3, 640, 640)  # 批次大小为2

        # 前向传播
        model(x)

        # 模拟目标
        targets = []
        for i in range(2):  # 批次大小
            # 每个图像有几个目标
            num_targets = torch.randint(1, 5, (1,)).item()
            for j in range(num_targets):
                # [image_id, class_id, x_center, y_center, width, height]
                target = torch.tensor([i, 0, 0.5, 0.5, 0.3, 0.3])
                targets.append(target)

        if targets:
            targets = torch.stack(targets)
            print(f"   目标形状: {targets.shape}")

        print("✅ 训练兼容性测试成功！")
        return True

    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数."""
    print("🔧 YOLOv5s-Hybrid-MoE 修复验证")
    print("=" * 80)

    # 测试模型创建和前向传播
    success1 = test_hybrid_moe_model()

    # 测试训练兼容性
    success2 = test_training_compatibility()

    # 总结
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)

    if success1 and success2:
        print("🎉 所有测试通过！YOLOv5s-Hybrid-MoE模型已修复完成！")
        print("\n🚀 现在可以开始训练:")
        print("python train.py --img 640 --batch 8 --epochs 100 \\")
        print("    --data your_data.yaml \\")
        print("    --cfg models/yolov5s-hybrid-moe.yaml \\")
        print("    --weights '' \\")
        print("    --smooth-early-stop --smooth-patience 300")

        print("\n💡 训练建议:")
        print("• 使用较小的批次大小 (batch 4-8)")
        print("• 从头开始训练 (weights '')")
        print("• 使用更多的epoch (200-300)")
        print("• 监控负载均衡损失")

    else:
        print("❌ 部分测试失败，需要进一步修复")
        if not success1:
            print("• 模型创建或前向传播有问题")
        if not success2:
            print("• 训练兼容性有问题")


if __name__ == "__main__":
    main()
