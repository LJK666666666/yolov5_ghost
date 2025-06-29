#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有best.pt或last.pt模型并保存详细结果
包括Precision, Recall, mAP, 预测错误的图片等
支持通过命令行参数选择模型类型
"""

import os
import json
import shutil
import subprocess
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def get_available_train_folders():
    """获取所有可用的训练文件夹"""
    runs_dir = Path("runs")
    train_folders = []

    if runs_dir.exists():
        for folder in runs_dir.iterdir():
            if (folder.is_dir() and
                folder.name.startswith('train') and
                'epoch' in folder.name and
                not '_test_' in folder.name):  # 排除测试结果文件夹
                # 检查是否包含模型文件
                has_models = False
                for exp_dir in folder.iterdir():
                    if exp_dir.is_dir():
                        weights_dir = exp_dir / "weights"
                        if (weights_dir.exists() and
                            (weights_dir / "best.pt").exists() or (weights_dir / "last.pt").exists()):
                            has_models = True
                            break
                if has_models:
                    train_folders.append(folder.name)

    return sorted(train_folders)

def get_all_models(model_type='best', train_folder='train200epoch'):
    """获取所有指定类型的模型路径

    Args:
        model_type (str): 模型类型，'best' 或 'last'
        train_folder (str): 训练文件夹名称，如 'train200epoch', 'train300epoch'

    Returns:
        list: 包含模型信息的列表
    """
    train_dir = Path(f"runs/{train_folder}")
    models = []

    if not train_dir.exists():
        print(f"警告: 训练目录不存在 {train_dir}")
        return models

    model_filename = f"{model_type}.pt"

    for exp_dir in train_dir.iterdir():
        if exp_dir.is_dir():
            weights_dir = exp_dir / "weights"
            model_pt = weights_dir / model_filename
            if model_pt.exists():
                models.append({
                    'name': exp_dir.name,
                    'path': str(model_pt),
                    'type': model_type,
                    'train_folder': train_folder
                })

    return models

def parse_val_output(output_text):
    """从val.py的输出中解析性能指标"""
    results = {}

    # 查找包含性能指标的行
    lines = output_text.split('\n')

    # 调试：保存原始输出用于分析
    results['_debug_output'] = output_text

    for line in lines:
        # 查找包含 "all" 和性能指标的行，去除ANSI颜色代码
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)  # 移除ANSI颜色代码

        # 查找包含 "all" 的行，格式应该是: "all 779 1648 0.87 0.851 0.887 0.536"
        if clean_line.strip().startswith('all'):
            # 分割行并提取数值
            parts = clean_line.strip().split()
            if len(parts) >= 7:  # all + 6个数值
                try:
                    results['precision'] = float(parts[3])
                    results['recall'] = float(parts[4])
                    results['map50'] = float(parts[5])
                    results['map50_95'] = float(parts[6])
                except (ValueError, IndexError) as e:
                    continue

        # 查找NO-Safety Vest类别的指标
        # 实际格式: "        NO-Safety Vest        779        361      0.846      0.798      0.834      0.403"
        elif 'NO-Safety Vest' in clean_line:
            parts = clean_line.strip().split()
            if len(parts) >= 7:  # NO-Safety + Vest + 图片数 + 实例数 + 4个指标
                try:
                    # 格式: "NO-Safety Vest 779 361 0.846 0.798 0.834 0.403"
                    # 索引:     0       1   2   3     4     5     6     7
                    results['no_safety_vest_precision'] = float(parts[4])
                    results['no_safety_vest_recall'] = float(parts[5])
                    results['no_safety_vest_map50'] = float(parts[6])
                    results['no_safety_vest_map50_95'] = float(parts[7])
                    results['_debug_no_vest_line'] = clean_line
                    results['_debug_no_vest_parts'] = parts
                except (ValueError, IndexError) as e:
                    continue

    return results

def run_validation(model_info, output_dir, conf_thres=0.001, iou_thres=0.6):
    """运行验证并保存结果

    Args:
        model_info (dict): 模型信息字典
        output_dir (Path): 输出目录
        conf_thres (float): 置信度阈值
        iou_thres (float): IoU阈值
    """
    model_name = model_info['name']
    model_path = model_info['path']

    print(f"正在测试模型: {model_name}")

    # 运行验证命令 - 使用val模式而不是test模式
    cmd = [
        "python", "val.py",
        "--weights", model_path,
        "--data", "data/SafetyVests.v6/data.yaml",
        "--img", "640",
        "--batch", "16",
        "--conf", str(conf_thres),
        "--iou", str(iou_thres),
        "--task", "val",  # 改为val模式
        "--save-txt",
        "--save-conf",
        "--save-json",
        "--project", str(output_dir),
        "--name", model_name,
        "--exist-ok"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"模型 {model_name} 验证完成")
        
        # 合并stdout和stderr的输出
        full_output = result.stdout + result.stderr
        
        # 解析输出中的性能指标
        metrics = parse_val_output(full_output)
        return True, full_output, metrics
    except subprocess.CalledProcessError as e:
        print(f"模型 {model_name} 验证失败: {e}")
        return False, e.stderr, {}

def parse_results(model_name, output_dir):
    """解析验证结果"""
    results_dir = Path(output_dir) / model_name
    
    # 读取results.csv
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        # 获取最后一行的结果（最新的验证结果）
        latest_results = df.iloc[-1].to_dict()
        return latest_results
    else:
        print(f"警告: 未找到结果文件 {results_csv}")
        return None

def save_error_images(model_name, output_dir, base_output_dir):
    """保存预测错误的图片"""
    results_dir = Path(output_dir) / model_name
    error_dir = Path(base_output_dir) / "error_images" / model_name
    error_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集路径
    dataset_path = Path("data/SafetyVests.v6/valid")
    labels_path = dataset_path / "labels"
    
    # 获取所有预测结果文件
    pred_labels_dir = results_dir / "labels"
    if not pred_labels_dir.exists():
        print(f"警告: 预测标签目录不存在 {pred_labels_dir}")
        return
    
    pred_files = list(pred_labels_dir.glob("*.txt"))
    error_images = set()
    
    print(f"正在分析 {len(pred_files)} 个预测结果文件...")
    
    for pred_file in pred_files:
        # 获取对应的真实标签文件
        gt_file = labels_path / pred_file.name
        
        if not gt_file.exists():
            print(f"警告: 未找到对应的真实标签文件 {gt_file}")
            continue
        
        # 读取预测结果
        pred_boxes = []
        try:
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        conf = float(parts[5]) if len(parts) > 5 else 1.0
                        pred_boxes.append([class_id, x_center, y_center, width, height, conf])
        except Exception as e:
            print(f"读取预测文件失败 {pred_file}: {e}")
            continue
        
        # 读取真实标签
        gt_boxes = []
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        gt_boxes.append([class_id, x_center, y_center, width, height])
        except Exception as e:
            print(f"读取真实标签文件失败 {gt_file}: {e}")
            continue
        
        # 检查是否有预测错误
        has_error = False
        
        # 检查假阳性（预测了但实际没有的框）
        if len(pred_boxes) > len(gt_boxes):
            has_error = True
        
        # 检查假阴性（实际有但预测没有的框）
        elif len(pred_boxes) < len(gt_boxes):
            has_error = True
        
        # 检查类别错误或位置错误
        else:
            # 简单的IoU检查（这里使用简化的检查方法）
            for pred_box in pred_boxes:
                pred_class = pred_box[0]
                pred_center_x, pred_center_y = pred_box[1], pred_box[2]
                pred_w, pred_h = pred_box[3], pred_box[4]
                
                # 检查是否有匹配的真实框
                matched = False
                for gt_box in gt_boxes:
                    gt_class = gt_box[0]
                    gt_center_x, gt_center_y = gt_box[1], gt_box[2]
                    gt_w, gt_h = gt_box[3], gt_box[4]
                    
                    # 检查类别是否匹配
                    if pred_class == gt_class:
                        # 检查位置是否接近（简化的IoU检查）
                        center_dist = ((pred_center_x - gt_center_x) ** 2 + 
                                     (pred_center_y - gt_center_y) ** 2) ** 0.5
                        size_diff = abs(pred_w * pred_h - gt_w * gt_h)
                        
                        # 如果中心点距离小于阈值且大小差异不大，认为是匹配的
                        if center_dist < 0.1 and size_diff < 0.1:
                            matched = True
                            break
                
                if not matched:
                    has_error = True
                    break
        
        # 如果有错误，将对应的图片添加到错误图片列表
        if has_error:
            # 从文件名推断原始图片名
            # 预测文件名格式: original_name_jpg.rf.hash.txt
            # 需要找到对应的图片文件
            base_name = pred_file.stem  # 去掉.txt
            # 查找对应的图片文件
            image_extensions = ['.jpg', '.jpeg', '.png']
            for ext in image_extensions:
                # 尝试不同的图片文件名模式
                possible_names = [
                    base_name + ext,
                    base_name.replace('_jpg.rf.', '.') + ext,
                    base_name.split('_jpg.rf.')[0] + ext
                ]
                
                for possible_name in possible_names:
                    # 检查在数据集目录中是否有对应的图片
                    dataset_image = dataset_path / "images" / possible_name
                    if dataset_image.exists():
                        error_images.add(dataset_image)
                        break
    
    # 复制错误图片到error_images目录
    copied_count = 0
    for img_path in error_images:
        try:
            # 复制图片
            shutil.copy2(img_path, error_dir / img_path.name)
            copied_count += 1
        except Exception as e:
            print(f"复制图片失败 {img_path}: {e}")
    
    print(f"已保存 {copied_count} 张预测错误的图片到 {error_dir}")
    
    # 如果没有找到错误图片，创建一个说明文件
    if copied_count == 0:
        with open(error_dir / "no_errors_found.txt", 'w', encoding='utf-8') as f:
            f.write("未发现预测错误的图片\n")
            f.write("所有预测结果都是正确的\n")
        print("未发现预测错误，已创建说明文件")

def create_summary_report(models_results, output_dir, model_type='best', train_folder='train200epoch'):
    """创建汇总报告

    Args:
        models_results (dict): 模型结果字典
        output_dir (Path): 输出目录
        model_type (str): 模型类型，'best' 或 'last'
        train_folder (str): 训练文件夹名称
    """
    summary_file = Path(output_dir) / "summary_report.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"YOLOv5 {model_type.upper()}模型测试汇总报告\n")
        f.write("=" * 100 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试数据集: data/SafetyVests.v6/valid\n")
        f.write(f"训练文件夹: runs/{train_folder}\n")
        f.write(f"模型类型: {model_type}.pt\n")
        f.write(f"模型数量: {len(models_results)}\n\n")

        # 创建整体性能结果表格
        f.write("整体模型性能对比:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'模型名称':<20} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}\n")
        f.write("-" * 100 + "\n")

        for model_name, results in models_results.items():
            if results:
                precision = results.get('precision', 'N/A')
                recall = results.get('recall', 'N/A')
                map50 = results.get('map50', 'N/A')
                map50_95 = results.get('map50_95', 'N/A')

                if isinstance(precision, (int, float)):
                    f.write(f"{model_name:<20} {precision:<12.4f} {recall:<12.4f} {map50:<12.4f} {map50_95:<15.4f}\n")
                else:
                    f.write(f"{model_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}\n")
            else:
                f.write(f"{model_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}\n")

        f.write("-" * 100 + "\n\n")

        # 创建NO-Safety Vest类别专门的性能表格
        f.write("NO-Safety Vest 类别性能对比:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'模型名称':<20} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}\n")
        f.write("-" * 100 + "\n")

        for model_name, results in models_results.items():
            if results:
                no_vest_precision = results.get('no_safety_vest_precision', 'N/A')
                no_vest_recall = results.get('no_safety_vest_recall', 'N/A')
                no_vest_map50 = results.get('no_safety_vest_map50', 'N/A')
                no_vest_map50_95 = results.get('no_safety_vest_map50_95', 'N/A')

                if isinstance(no_vest_precision, (int, float)):
                    f.write(f"{model_name:<20} {no_vest_precision:<12.4f} {no_vest_recall:<12.4f} {no_vest_map50:<12.4f} {no_vest_map50_95:<15.4f}\n")
                else:
                    f.write(f"{model_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}\n")
            else:
                f.write(f"{model_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}\n")

        f.write("-" * 100 + "\n\n")

        # 找出最佳模型（基于整体mAP@0.5）
        best_model_overall = None
        best_map50_overall = 0

        for model_name, results in models_results.items():
            if results:
                map50 = results.get('map50', 0)
                if isinstance(map50, (int, float)) and map50 > best_map50_overall:
                    best_map50_overall = map50
                    best_model_overall = model_name

        # 找出NO-Safety Vest召回率最佳的模型
        best_model_no_vest_recall = None
        best_no_vest_recall = 0

        for model_name, results in models_results.items():
            if results:
                no_vest_recall = results.get('no_safety_vest_recall', 0)
                if isinstance(no_vest_recall, (int, float)) and no_vest_recall > best_no_vest_recall:
                    best_no_vest_recall = no_vest_recall
                    best_model_no_vest_recall = model_name

        f.write("最佳模型分析:\n")
        f.write("-" * 50 + "\n")
        if best_model_overall:
            f.write(f"整体最佳模型 (基于mAP@0.5): {best_model_overall}\n")
            f.write(f"最佳整体mAP@0.5: {best_map50_overall:.4f}\n\n")

        if best_model_no_vest_recall:
            f.write(f"NO-Safety Vest召回率最佳模型: {best_model_no_vest_recall}\n")
            f.write(f"最佳NO-Safety Vest召回率: {best_no_vest_recall:.4f}\n\n")

        f.write("详细结果文件位置:\n")
        for model_name in models_results.keys():
            f.write(f"- {model_name}: {output_dir}/{model_name}/\n")

        f.write(f"\n错误图片位置: {output_dir}/error_images/\n")

    print(f"汇总报告已保存到: {summary_file}")

def parse_args():
    """解析命令行参数"""
    # 获取可用的训练文件夹
    available_folders = get_available_train_folders()

    parser = argparse.ArgumentParser(
        description='测试YOLOv5模型性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用的训练文件夹:
{chr(10).join(f'  - {folder}' for folder in available_folders) if available_folders else '  未找到训练文件夹'}

使用示例:
  python test_all_models.py --model-type best --train-folder train200epoch
  python test_all_models.py --model-type last --train-folder train300epoch --conf-thres 0.25
        """
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['best', 'last'],
        default='best',
        help='选择模型类型: best.pt 或 last.pt (默认: best)'
    )
    parser.add_argument(
        '--train-folder',
        type=str,
        default='train200epoch',
        help=f'训练文件夹名称 (默认: train200epoch)\n可用选项: {", ".join(available_folders) if available_folders else "无"}'
    )
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.001,
        help='置信度阈值 (默认: 0.001)'
    )
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.6,
        help='IoU阈值 (默认: 0.6)'
    )
    parser.add_argument(
        '--list-folders',
        action='store_true',
        help='列出所有可用的训练文件夹并退出'
    )

    args = parser.parse_args()

    # 如果用户要求列出文件夹，显示后退出
    if args.list_folders:
        print("可用的训练文件夹:")
        if available_folders:
            for folder in available_folders:
                print(f"  - {folder}")
        else:
            print("  未找到任何训练文件夹")
        sys.exit(0)

    # 验证训练文件夹是否存在
    if args.train_folder not in available_folders:
        print(f"错误: 训练文件夹 '{args.train_folder}' 不存在")
        print(f"可用的训练文件夹: {', '.join(available_folders) if available_folders else '无'}")
        sys.exit(1)

    return args

def main():
    """主函数"""
    args = parse_args()
    model_type = args.model_type
    train_folder = args.train_folder

    print(f"开始测试 {train_folder} 文件夹下的所有{model_type}.pt模型...")

    # 获取所有模型
    models = get_all_models(model_type, train_folder)
    if not models:
        print(f"未找到任何{model_type}.pt模型文件在 runs/{train_folder} 目录下！")
        print(f"请检查目录是否存在以及是否包含模型文件。")
        return

    print(f"找到 {len(models)} 个{model_type}模型:")
    for model in models:
        print(f"  - {model['name']}: {model['path']}")

    # 创建输出目录 - 包含训练文件夹名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/{train_folder}_test_{model_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n训练文件夹: runs/{train_folder}")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {args.conf_thres}")
    print(f"IoU阈值: {args.iou_thres}")

    # 存储所有模型的结果
    models_results = {}

    # 测试每个模型
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 测试模型: {model['name']}")
        success, output, metrics = run_validation(model, output_dir, args.conf_thres, args.iou_thres)

        if success:
            # 保存解析的指标
            models_results[model['name']] = metrics

            # 保存错误图片
            save_error_images(model['name'], output_dir, output_dir)

            print(f"模型 {model['name']} 测试完成")
            if metrics:
                print(f"  整体性能指标: Precision={metrics.get('precision', 'N/A')}, "
                      f"Recall={metrics.get('recall', 'N/A')}, "
                      f"mAP@0.5={metrics.get('map50', 'N/A')}, "
                      f"mAP@0.5:0.95={metrics.get('map50_95', 'N/A')}")

                # 显示NO-Safety Vest类别的召回率
                no_vest_recall = metrics.get('no_safety_vest_recall', 'N/A')
                if no_vest_recall != 'N/A':
                    print(f"  NO-Safety Vest召回率: {no_vest_recall:.4f}")
                    print(f"  NO-Safety Vest其他指标: Precision={metrics.get('no_safety_vest_precision', 'N/A'):.4f}, "
                          f"mAP@0.5={metrics.get('no_safety_vest_map50', 'N/A'):.4f}, "
                          f"mAP@0.5:0.95={metrics.get('no_safety_vest_map50_95', 'N/A'):.4f}")
                else:
                    print(f"  NO-Safety Vest召回率: 未能解析")
                    # 调试信息
                    if '_debug_no_vest_line' in metrics:
                        print(f"  调试: 找到NO-Safety Vest行: {metrics['_debug_no_vest_line']}")
                    elif '_debug_no_vest_line_fallback' in metrics:
                        print(f"  调试: 使用备选解析: {metrics['_debug_no_vest_line_fallback']}")
                    else:
                        print(f"  调试: 未找到NO-Safety Vest相关行，请检查输出格式")
        else:
            print(f"模型 {model['name']} 测试失败")
            models_results[model['name']] = {}

    # 创建汇总报告
    create_summary_report(models_results, output_dir, model_type, train_folder)

    # 保存详细结果到JSON
    json_file = output_dir / "detailed_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(models_results, f, indent=2, ensure_ascii=False)

    print(f"\n测试完成！")
    print(f"所有结果已保存到: {output_dir}")
    print(f"汇总报告: {output_dir}/summary_report.txt")
    print(f"详细JSON结果: {json_file}")

if __name__ == "__main__":
    main()