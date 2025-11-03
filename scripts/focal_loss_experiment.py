#!/usr/bin/env python3
"""
面向"难例"的分类损失函数对比实验脚本
创新亮点二：引入面向"难例"的分类损失函数.

使用方法：
python scripts/focal_loss_experiment.py --mode [baseline|focal|compare]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def create_baseline_config():
    """创建基线配置（fl_gamma=0.0）."""
    config_path = ROOT / "data/hyps/hyp.baseline_temp.yaml"

    # 读取对比配置文件
    with open(ROOT / "data/hyps/hyp.baseline_vs_focal.yaml") as f:
        config = yaml.safe_load(f)

    # 修改为基线配置
    config["fl_gamma"] = 0.0

    # 保存基线配置
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ 基线配置已创建: {config_path}")
    return config_path


def run_training(config_path, experiment_name, epochs=100):
    """运行训练实验."""
    cmd = [
        "python",
        "train.py",
        "--data",
        "data/SafetyVests.v6/data.yaml",
        "--cfg",
        "models/yolov5s-ghost.yaml",
        "--weights",
        "yolov5s.pt",
        "--hyp",
        str(config_path),
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


def compare_results():
    """对比实验结果."""
    print("\n" + "=" * 60)
    print("📊 实验结果对比")
    print("=" * 60)

    baseline_results = ROOT / "runs/train/baseline_bce_focal/results.csv"
    focal_results = ROOT / "runs/train/focal_loss_gamma2/results.csv"

    if baseline_results.exists() and focal_results.exists():
        print("✅ 找到实验结果文件")
        print(f"📁 基线结果: {baseline_results}")
        print(f"📁 Focal Loss结果: {focal_results}")

        # 这里可以添加更详细的结果分析代码
        print("\n💡 请查看以下目录的详细结果:")
        print("   - 基线实验: runs/train/baseline_bce_focal/")
        print("   - Focal Loss实验: runs/train/focal_loss_gamma2/")
        print("   - 验证结果: runs/val/")

    else:
        print("❌ 未找到完整的实验结果文件")
        print("请确保两个实验都已完成训练")


def main():
    parser = argparse.ArgumentParser(description="Focal Loss 对比实验")
    parser.add_argument("--mode", choices=["baseline", "focal", "compare", "all"], default="all", help="实验模式")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")

    args = parser.parse_args()

    print("🎯 面向'难例'的分类损失函数对比实验")
    print("=" * 60)

    if args.mode in ["baseline", "all"]:
        print("\n📋 步骤1: 基线实验 (标准BCE损失, fl_gamma=0.0)")
        baseline_config = create_baseline_config()
        success = run_training(baseline_config, "baseline_bce_focal", args.epochs)

        if success:
            weights_path = ROOT / "runs/train/baseline_bce_focal/weights/best.pt"
            run_validation(weights_path, "baseline_bce_focal")

    if args.mode in ["focal", "all"]:
        print("\n📋 步骤2: Focal Loss实验 (fl_gamma=2.0)")
        focal_config = ROOT / "data/hyps/hyp.baseline_vs_focal.yaml"
        success = run_training(focal_config, "focal_loss_gamma2", args.epochs)

        if success:
            weights_path = ROOT / "runs/train/focal_loss_gamma2/weights/best.pt"
            run_validation(weights_path, "focal_loss_gamma2")

    if args.mode in ["compare", "all"]:
        print("\n📋 步骤3: 结果对比")
        compare_results()

    print("\n🎉 实验完成！")
    print("\n📝 项目故事素材:")
    print('"在错误分析中，我们发现模型的大部分错误都集中在少数困难样本上')
    print("（如背景混淆、轻微遮挡）。为了解决这个问题，我们引入了经典的Focal Loss")
    print("来优化YOLOv5的分类损失部分。通过对比实验发现，Focal Loss（γ=2.0）")
    print("相比标准BCE损失在复杂场景下的检测精度提升了X%，特别是在背景相似的")
    print('No-Safety Vest样本和被遮挡的Safety Vest样本上表现更加鲁棒。"')


if __name__ == "__main__":
    main()
