#!/usr/bin/env python3
"""
加权特征融合对比实验脚本 创新亮点三：构建"加权"的特征融合颈部网络.

使用方法： python scripts/weighted_feature_fusion_experiment.py --mode [baseline|wff|wff_concat|compare|all]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def test_model_loading():
    """测试模型是否能正确加载."""
    print("🔍 测试模型加载...")

    try:
        # 测试基线模型
        from models.yolo import Model

        model_baseline = Model(ROOT / "models/yolov5s.yaml")
        print(f"✅ 基线模型加载成功: {sum(p.numel() for p in model_baseline.parameters())} 参数")

        # 测试WFF模型
        model_wff = Model(ROOT / "models/yolov5s-wff.yaml")
        print(f"✅ WFF模型加载成功: {sum(p.numel() for p in model_wff.parameters())} 参数")

        # 测试WFF-Concat模型
        model_wff_concat = Model(ROOT / "models/yolov5s-wff-concat.yaml")
        print(f"✅ WFF-Concat模型加载成功: {sum(p.numel() for p in model_wff_concat.parameters())} 参数")

        # 参数对比
        baseline_params = sum(p.numel() for p in model_baseline.parameters())
        wff_params = sum(p.numel() for p in model_wff.parameters())
        wff_concat_params = sum(p.numel() for p in model_wff_concat.parameters())

        print("\n📊 参数对比:")
        print(f"   基线模型:     {baseline_params:,} 参数")
        print(f"   WFF模型:      {wff_params:,} 参数 (+{wff_params - baseline_params:,})")
        print(f"   WFF-Concat:   {wff_concat_params:,} 参数 (+{wff_concat_params - baseline_params:,})")

        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


def run_training(config_path, experiment_name, epochs=100):
    """运行训练实验."""
    cmd = [
        "python",
        "train.py",
        "--data",
        "data/SafetyVests.v6/data.yaml",
        "--cfg",
        str(config_path),
        "--weights",
        "yolov5s.pt",
        "--name",
        experiment_name,
        "--epochs",
        str(epochs),
        "--save-period",
        "10",  # 每10个epoch保存一次
    ]

    print(f"🚀 开始训练实验: {experiment_name}")
    print(f"📝 命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"✅ 训练完成: {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {experiment_name}, 错误: {e}")
        return False


def run_validation(weights_path, experiment_name):
    """运行验证实验."""
    cmd = [
        "python",
        "val.py",
        "--data",
        "data/SafetyVests.v6/data.yaml",
        "--weights",
        str(weights_path),
        "--name",
        f"{experiment_name}_val",
        "--save-txt",
        "--save-conf",
    ]

    print(f"🔍 开始验证实验: {experiment_name}")
    print(f"📝 命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"✅ 验证完成: {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 验证失败: {experiment_name}, 错误: {e}")
        return False


def analyze_weights():
    """分析加权特征融合的权重学习情况."""
    print("\n🔍 分析加权特征融合权重...")

    try:
        # 加载训练好的模型
        wff_weights = ROOT / "runs/train/wff_experiment/weights/best.pt"
        wff_concat_weights = ROOT / "runs/train/wff_concat_experiment/weights/best.pt"

        if wff_weights.exists():
            checkpoint = torch.load(wff_weights, map_location="cpu")
            model_state = checkpoint["model"].state_dict()

            print("📊 WFF模型的学习权重:")
            for name, param in model_state.items():
                if "weights" in name and "WeightedFeatureFusion" in name:
                    weights = param.cpu().numpy()
                    normalized_weights = weights / weights.sum()
                    print(f"   {name}: {normalized_weights}")

        if wff_concat_weights.exists():
            checkpoint = torch.load(wff_concat_weights, map_location="cpu")
            model_state = checkpoint["model"].state_dict()

            print("📊 WFF-Concat模型的学习权重:")
            for name, param in model_state.items():
                if "weights" in name and "WeightedFeatureFusion" in name:
                    weights = param.cpu().numpy()
                    normalized_weights = weights / weights.sum()
                    print(f"   {name}: {normalized_weights}")

    except Exception as e:
        print(f"❌ 权重分析失败: {e}")


def compare_results():
    """对比实验结果."""
    print("\n" + "=" * 60)
    print("📊 加权特征融合实验结果对比")
    print("=" * 60)

    experiments = [
        ("baseline_wff", "基线模型"),
        ("wff_experiment", "WFF模型"),
        ("wff_concat_experiment", "WFF-Concat模型"),
    ]

    for exp_name, exp_desc in experiments:
        results_file = ROOT / f"runs/train/{exp_name}/results.csv"
        if results_file.exists():
            print(f"✅ {exp_desc}: {results_file}")
        else:
            print(f"❌ {exp_desc}: 结果文件不存在")

    print("\n💡 请查看以下目录的详细结果:")
    for exp_name, exp_desc in experiments:
        print(f"   - {exp_desc}: runs/train/{exp_name}/")
    print("   - 验证结果: runs/val/")

    # 分析权重
    analyze_weights()


def main():
    parser = argparse.ArgumentParser(description="加权特征融合对比实验")
    parser.add_argument(
        "--mode", choices=["baseline", "wff", "wff_concat", "compare", "test", "all"], default="test", help="实验模式"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")

    args = parser.parse_args()

    print("🎯 构建'加权'的特征融合颈部网络实验")
    print("=" * 60)

    if args.mode in ["test", "all"]:
        print("\n📋 步骤0: 测试模型加载")
        if not test_model_loading():
            print("❌ 模型加载测试失败，请检查代码")
            return

    if args.mode in ["baseline", "all"]:
        print("\n📋 步骤1: 基线实验 (标准YOLOv5s)")
        success = run_training(ROOT / "models/yolov5s.yaml", "baseline_wff", args.epochs)

        if success:
            weights_path = ROOT / "runs/train/baseline_wff/weights/best.pt"
            run_validation(weights_path, "baseline_wff")

    if args.mode in ["wff", "all"]:
        print("\n📋 步骤2: WFF实验 (加权特征融合)")
        success = run_training(ROOT / "models/yolov5s-wff.yaml", "wff_experiment", args.epochs)

        if success:
            weights_path = ROOT / "runs/train/wff_experiment/weights/best.pt"
            run_validation(weights_path, "wff_experiment")

    if args.mode in ["wff_concat", "all"]:
        print("\n📋 步骤3: WFF-Concat实验 (加权特征融合拼接)")
        success = run_training(ROOT / "models/yolov5s-wff-concat.yaml", "wff_concat_experiment", args.epochs)

        if success:
            weights_path = ROOT / "runs/train/wff_concat_experiment/weights/best.pt"
            run_validation(weights_path, "wff_concat_experiment")

    if args.mode in ["compare", "all"]:
        print("\n📋 步骤4: 结果对比")
        compare_results()

    print("\n🎉 实验完成！")
    print("\n📝 项目故事素材:")
    print('"我们发现，标准的YOLOv5在进行特征融合时，对所有尺度的特征图一视同仁。')
    print("我们认为，对于反光衣检测这类任务，某些特定尺度的特征可能更为关键。")
    print("因此，我们借鉴了BiFPN的核心思想，引入了加权融合机制。实验结果表明，")
    print("加权特征融合相比标准拼接在复杂场景下的检测精度提升了X%，特别是在")
    print('多尺度目标检测上表现更加出色。"')


if __name__ == "__main__":
    main()
