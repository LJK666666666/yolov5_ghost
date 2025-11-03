#!/usr/bin/env python3
"""使用safety-helmet-vest数据集训练多个YOLOv5模型的脚本."""

import subprocess
import sys
import time
from pathlib import Path


def run_training_command(cmd, model_name):
    """运行训练命令并处理结果."""
    print(f"\n{'=' * 60}")
    print(f"开始训练模型: {model_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # 运行训练命令
        subprocess.run(cmd, check=True, capture_output=False)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ 模型 {model_name} 训练成功!")
        print(f"训练时间: {duration / 3600:.2f} 小时")
        return True

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n❌ 模型 {model_name} 训练失败!")
        print(f"错误代码: {e.returncode}")
        print(f"运行时间: {duration / 60:.2f} 分钟")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ 模型 {model_name} 训练被用户中断!")
        return False


def main():
    """主函数 - 依次训练所有模型."""
    # 检查数据集是否存在
    data_yaml = Path("data/safety-helmet-vest/data.yaml")
    if not data_yaml.exists():
        print("❌ 错误: 数据集配置文件不存在!")
        print("请先运行 python download_safety_helmet_dataset.py 下载并准备数据集")
        sys.exit(1)

    # 检查模型配置文件是否存在
    model_configs = [
        "models/yolov5s.yaml",
        "models/yolov5s-ghost_12.yaml",
        "models/yolov5s-ghost_1.yaml",
        "models/yolov5s-ghost_2.yaml",
    ]

    for config in model_configs:
        if not Path(config).exists():
            print(f"❌ 错误: 模型配置文件不存在: {config}")
            sys.exit(1)

    # 训练配置
    base_args = [
        "python",
        "train.py",
        "--data",
        str(data_yaml),
        "--weights",
        "yolov5s.pt",
        "--project",
        "runs/safety_helmet_train300epoch",
        "--epochs",
        "300",
        "--patience",
        "100",
        "--batch-size",
        "32",
    ]

    # 定义所有训练任务
    training_tasks = [
        {"name": "YOLOv5s Baseline", "cmd": base_args + ["--cfg", "models/yolov5s.yaml", "--name", "yolov5s_"]},
        {
            "name": "YOLOv5s-Ghost_123 + WIoU",
            "cmd": base_args
            + ["--cfg", "models/yolov5s-ghost_12.yaml", "--name", "yolov5s-ghost_123_", "--box-loss", "wiou"],
        },
        {
            "name": "YOLOv5s-Ghost_1",
            "cmd": base_args + ["--cfg", "models/yolov5s-ghost_1.yaml", "--name", "yolov5s-ghost_1_"],
        },
        {
            "name": "YOLOv5s-Ghost_2",
            "cmd": base_args + ["--cfg", "models/yolov5s-ghost_2.yaml", "--name", "yolov5s-ghost_2_"],
        },
        {
            "name": "YOLOv5s + WIoU",
            "cmd": base_args + ["--cfg", "models/yolov5s.yaml", "--name", "yolov5s-ghost_3_", "--box-loss", "wiou"],
        },
    ]

    # 记录训练结果
    results = []
    total_start_time = time.time()

    print(f"🚀 开始训练 {len(training_tasks)} 个模型")
    print(f"数据集: {data_yaml}")
    print("输出目录: runs/safety_helmet_train300epoch")
    print("批次大小: 32")
    print("训练轮数: 300")

    # 依次训练每个模型
    for i, task in enumerate(training_tasks, 1):
        print(f"\n📊 进度: {i}/{len(training_tasks)}")

        success = run_training_command(task["cmd"], task["name"])
        results.append({"name": task["name"], "success": success})

        # 如果训练失败，询问是否继续
        if not success:
            response = input(f"\n模型 {task['name']} 训练失败，是否继续训练下一个模型? (y/n): ")
            if response.lower() != "y":
                print("训练被用户终止")
                break

    # 输出最终结果
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"\n{'=' * 60}")
    print("🎯 训练完成总结")
    print(f"{'=' * 60}")
    print(f"总训练时间: {total_duration / 3600:.2f} 小时")
    print(f"成功训练: {sum(1 for r in results if r['success'])}/{len(results)} 个模型")

    print("\n📋 详细结果:")
    for result in results:
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"  {result['name']}: {status}")

    # 显示结果目录
    results_dir = Path("runs/safety_helmet_train300epoch")
    if results_dir.exists():
        print(f"\n📁 训练结果保存在: {results_dir}")
        print("可用的模型:")
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                weights_dir = model_dir / "weights"
                if weights_dir.exists():
                    best_pt = weights_dir / "best.pt"
                    last_pt = weights_dir / "last.pt"
                    if best_pt.exists() or last_pt.exists():
                        print(f"  - {model_dir.name}")


if __name__ == "__main__":
    main()
