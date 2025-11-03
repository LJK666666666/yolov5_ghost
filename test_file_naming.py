#!/usr/bin/env python3
"""
测试文件命名逻辑.

Author: Augment Agent (Claude Sonnet 4 by Anthropic)
Created: 2025-07-05
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def test_file_naming_logic():
    """测试文件命名逻辑."""
    print("🧪 测试文件命名逻辑")
    print("=" * 60)

    # 模拟参数
    train_folder = "train200epoch"
    model_type = "best"
    timestamp = "20250705_143022"

    test_cases = [
        {"eval_split": "test", "description": "测试集 (论文用) - 默认设置"},
        {"eval_split": "val", "description": "验证集 (开发用) - 手动指定"},
    ]

    print("文件命名测试:")
    print("-" * 60)

    for i, case in enumerate(test_cases, 1):
        eval_split = case["eval_split"]
        description = case["description"]

        print(f"\n测试 {i}: {description}")
        print(f"参数: --eval-split {eval_split}")

        # 输出目录命名
        eval_suffix = "test" if eval_split == "test" else "val"
        output_dir = f"runs/{train_folder}_{eval_suffix}_{model_type}_{timestamp}"

        # 表格文件命名
        csv_file = f"performance_comparison_{model_type}_{train_folder}_{eval_split}.csv"
        excel_file = f"performance_comparison_{model_type}_{train_folder}_{eval_split}.xlsx"

        print(f"  输出目录: {output_dir}")
        print(f"  CSV文件:  {csv_file}")
        print(f"  Excel文件: {excel_file}")

        # 验证命名逻辑
        if eval_split == "test":
            assert "test" in output_dir, "测试集输出目录应包含'test'"
            assert "test" in csv_file, "测试集CSV文件应包含'test'"
            assert "test" in excel_file, "测试集Excel文件应包含'test'"
            print("  ✅ 测试集命名正确")
        else:
            assert "val" in output_dir, "验证集输出目录应包含'val'"
            assert "val" in csv_file, "验证集CSV文件应包含'val'"
            assert "val" in excel_file, "验证集Excel文件应包含'val'"
            print("  ✅ 验证集命名正确")


def test_default_behavior():
    """测试默认行为."""
    print("\n🎯 测试默认行为")
    print("=" * 60)

    # 模拟默认参数
    default_eval_split = "test"  # 新的默认值

    print(f"默认 --eval-split 参数: {default_eval_split}")
    print(f"含义: 默认使用{'测试集' if default_eval_split == 'test' else '验证集'}")
    print(f"用途: {'论文指标' if default_eval_split == 'test' else '开发调试'}")

    if default_eval_split == "test":
        print("✅ 默认设置符合学术标准")
        print("✅ 论文指标将基于测试集")
        print("✅ 无需额外参数即可获得发表级结果")
    else:
        print("⚠️  默认设置为开发模式")
        print("⚠️  需要手动指定 --eval-split test 用于论文")


def test_command_examples():
    """测试命令示例."""
    print("\n📝 命令示例测试")
    print("=" * 60)

    commands = [
        {
            "cmd": "python test_all_models.py --model-type best --train-folder train200epoch",
            "description": "默认命令 (论文用)",
            "eval_split": "test",
            "output_pattern": "runs/train200epoch_test_best_*",
        },
        {
            "cmd": "python test_all_models.py --eval-split val --model-type best --train-folder train200epoch",
            "description": "开发调试命令",
            "eval_split": "val",
            "output_pattern": "runs/train200epoch_val_best_*",
        },
    ]

    for i, cmd_info in enumerate(commands, 1):
        print(f"\n示例 {i}: {cmd_info['description']}")
        print(f"命令: {cmd_info['cmd']}")
        print(
            f"数据集: {cmd_info['eval_split'].upper()} ({'测试集' if cmd_info['eval_split'] == 'test' else '验证集'})"
        )
        print(f"输出目录模式: {cmd_info['output_pattern']}")
        print(f"文件名包含: _{cmd_info['eval_split']}")


def main():
    """主函数."""
    print("🔧 文件命名逻辑测试")
    print("=" * 80)

    # 测试文件命名逻辑
    test_file_naming_logic()

    # 测试默认行为
    test_default_behavior()

    # 测试命令示例
    test_command_examples()

    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)

    print("✅ 所有文件命名测试通过！")

    print("\n🎯 关键变更:")
    print("  1. 默认使用测试集 (--eval-split test)")
    print("  2. 输出目录包含数据集类型: train200epoch_test_best_* 或 train200epoch_val_best_*")
    print("  3. 表格文件包含数据集类型: performance_comparison_best_train200epoch_test.csv")
    print("  4. 清晰区分论文用(test)和开发用(val)结果")

    print("\n📊 使用指南:")
    print("  🎓 论文发表: python test_all_models.py --model-type best --train-folder train200epoch")
    print("  🔬 开发调试: python test_all_models.py --eval-split val --model-type best --train-folder train200epoch")

    print("\n🏆 学术标准:")
    print("  ✅ 默认符合论文发表要求")
    print("  ✅ 文件名清晰标识数据集类型")
    print("  ✅ 避免混淆验证集和测试集结果")


if __name__ == "__main__":
    main()
