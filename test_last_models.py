#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有last.pt模型并保存详细结果
包括Precision, Recall, mAP, 预测错误的图片等
"""

import os
import json
import shutil
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def get_all_best_models():
    """获取所有best.pt模型路径"""
    train_dir = Path("runs/train200epoch")
    models = []
    
    for exp_dir in train_dir.iterdir():
        if exp_dir.is_dir():
            weights_dir = exp_dir / "weights"
            last_pt = weights_dir / "last.pt"
            if last_pt.exists():
                models.append({
                    'name': exp_dir.name,
                    'path': str(last_pt)
                })
    
    return models

def parse_val_output(output_text):
    """从val.py的输出中解析性能指标"""
    results = {}
    
    # 查找包含性能指标的行
    lines = output_text.split('\n')
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
                    break
                except (ValueError, IndexError) as e:
                    continue
    
    return results

def run_validation(model_info, output_dir):
    """运行验证并保存结果"""
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
        "--conf", "0.001",
        "--iou", "0.6",
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

def create_summary_report(models_results, output_dir):
    """创建汇总报告"""
    summary_file = Path(output_dir) / "summary_report.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLOv5 模型测试汇总报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试数据集: data/SafetyVests.v6/test\n")
        f.write(f"模型数量: {len(models_results)}\n\n")
        
        # 创建结果表格
        f.write("模型性能对比:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模型名称':<20} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}\n")
        f.write("-" * 80 + "\n")
        
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
        
        f.write("-" * 80 + "\n\n")
        
        # 找出最佳模型
        best_model = None
        best_map50 = 0
        
        for model_name, results in models_results.items():
            if results:
                map50 = results.get('map50', 0)
                if isinstance(map50, (int, float)) and map50 > best_map50:
                    best_map50 = map50
                    best_model = model_name
        
        if best_model:
            f.write(f"最佳模型 (基于mAP@0.5): {best_model}\n")
            f.write(f"最佳mAP@0.5: {best_map50:.4f}\n\n")
        
        f.write("详细结果文件位置:\n")
        for model_name in models_results.keys():
            f.write(f"- {model_name}: {output_dir}/{model_name}/\n")
        
        f.write(f"\n错误图片位置: {output_dir}/error_images/\n")
    
    print(f"汇总报告已保存到: {summary_file}")

def main():
    """主函数"""
    print("开始测试所有best.pt模型...")
    
    # 获取所有模型
    models = get_all_best_models()
    if not models:
        print("未找到任何best.pt模型文件！")
        return
    
    print(f"找到 {len(models)} 个模型:")
    for model in models:
        print(f"  - {model['name']}: {model['path']}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/last_test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_dir}")
    
    # 存储所有模型的结果
    models_results = {}
    
    # 测试每个模型
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 测试模型: {model['name']}")
        success, output, metrics = run_validation(model, output_dir)
        
        if success:
            # 保存解析的指标
            models_results[model['name']] = metrics
            
            # 保存错误图片
            save_error_images(model['name'], output_dir, output_dir)
            
            print(f"模型 {model['name']} 测试完成")
            if metrics:
                print(f"  性能指标: Precision={metrics.get('precision', 'N/A')}, "
                      f"Recall={metrics.get('recall', 'N/A')}, "
                      f"mAP@0.5={metrics.get('map50', 'N/A')}, "
                      f"mAP@0.5:0.95={metrics.get('map50_95', 'N/A')}")
        else:
            print(f"模型 {model['name']} 测试失败")
            models_results[model['name']] = {}
    
    # 创建汇总报告
    create_summary_report(models_results, output_dir)
    
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