#!/usr/bin/env python3
"""
快速查看预测分析结果的脚本
"""

import os
import json
import glob
from collections import Counter

def view_analysis_results(analysis_dir='test_prediction_analysis'):
    """查看分析结果"""
    
    print("=" * 60)
    print("YOLOv5模型预测分析结果概览")
    print("=" * 60)
    
    # 读取总体统计
    overall_stats_file = os.path.join(analysis_dir, 'overall_statistics.json')
    if os.path.exists(overall_stats_file):
        with open(overall_stats_file, 'r') as f:
            overall_stats = json.load(f)
        
        print("\n📊 总体性能统计:")
        print("-" * 40)
        for split_name, stats in overall_stats.items():
            accuracy = stats['accuracy'] * 100
            total_images = stats['total_images']
            correct = stats['correct_predictions']
            
            print(f"\n{split_name.upper()}:")
            print(f"  总图像数: {total_images:,}")
            print(f"  正确预测: {correct:,}")
            print(f"  准确率: {accuracy:.2f}%")
            
            # 错误统计
            errors = stats['error_statistics']
            if errors:
                print(f"  错误统计:")
                for error_type, count in errors.items():
                    error_names = {
                        'false_positives': '误检',
                        'false_negatives': '漏检',
                        'low_iou_matches': '低IoU',
                        'confidence_issues': '低置信度'
                    }
                    print(f"    {error_names.get(error_type, error_type)}: {count}")
    
    # 分析错误模式
    print(f"\n🔍 错误模式分析:")
    print("-" * 40)
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(analysis_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        print(f"\n{split.upper()} 错误详情:")
        
        # 分析误检
        fp_dir = os.path.join(split_dir, 'false_positives')
        if os.path.exists(fp_dir):
            fp_info_files = glob.glob(os.path.join(fp_dir, '*_info.json'))
            if fp_info_files:
                fp_classes = []
                fp_confidences = []
                
                for info_file in fp_info_files[:20]:  # 只分析前20个
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    for error in info['errors']:
                        if error['type'] == 'false_positive':
                            fp_classes.append(error['predicted_class'])
                            fp_confidences.append(error['confidence'])
                
                if fp_classes:
                    class_counts = Counter(fp_classes)
                    avg_conf = sum(fp_confidences) / len(fp_confidences)
                    print(f"  误检分析 ({len(fp_info_files)}个图像):")
                    print(f"    误检类别分布: {dict(class_counts)}")
                    print(f"    平均置信度: {avg_conf:.3f}")
        
        # 分析漏检
        fn_dir = os.path.join(split_dir, 'false_negatives')
        if os.path.exists(fn_dir):
            fn_info_files = glob.glob(os.path.join(fn_dir, '*_info.json'))
            if fn_info_files:
                fn_classes = []
                
                for info_file in fn_info_files[:20]:  # 只分析前20个
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    for error in info['errors']:
                        if error['type'] == 'false_negative':
                            fn_classes.append(error['true_class'])
                
                if fn_classes:
                    class_counts = Counter(fn_classes)
                    print(f"  漏检分析 ({len(fn_info_files)}个图像):")
                    print(f"    漏检类别分布: {dict(class_counts)}")
        
        # 分析低IoU
        low_iou_dir = os.path.join(split_dir, 'low_iou_matches')
        if os.path.exists(low_iou_dir):
            low_iou_info_files = glob.glob(os.path.join(low_iou_dir, '*_info.json'))
            if low_iou_info_files:
                ious = []
                confidences = []
                
                for info_file in low_iou_info_files[:20]:  # 只分析前20个
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    for error in info['errors']:
                        if error['type'] == 'low_iou' and 'iou' in error:
                            ious.append(error['iou'])
                            confidences.append(error['confidence'])
                
                if ious:
                    avg_iou = sum(ious) / len(ious)
                    avg_conf = sum(confidences) / len(confidences)
                    print(f"  低IoU分析 ({len(low_iou_info_files)}个图像):")
                    print(f"    平均IoU: {avg_iou:.3f}")
                    print(f"    平均置信度: {avg_conf:.3f}")
    
    # 改进建议
    print(f"\n💡 改进建议:")
    print("-" * 40)
    
    if overall_stats:
        # 计算总错误数
        total_errors = {}
        for stats in overall_stats.values():
            for error_type, count in stats['error_statistics'].items():
                total_errors[error_type] = total_errors.get(error_type, 0) + count
        
        # 按错误数量排序
        sorted_errors = sorted(total_errors.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n主要问题及建议:")
        for error_type, count in sorted_errors[:3]:  # 显示前3个主要问题
            if count > 0:
                error_names = {
                    'false_positives': '误检',
                    'false_negatives': '漏检',
                    'low_iou_matches': '低IoU匹配',
                    'confidence_issues': '低置信度'
                }
                
                suggestions = {
                    'false_positives': [
                        "提高置信度阈值",
                        "使用更严格的NMS参数",
                        "增加负样本训练",
                        "使用Focal Loss"
                    ],
                    'false_negatives': [
                        "降低置信度阈值",
                        "增加数据增强",
                        "使用多尺度训练",
                        "调整anchor尺寸"
                    ],
                    'low_iou_matches': [
                        "使用更精确的损失函数(DIoU, CIoU)",
                        "增加边界框回归权重",
                        "使用更高分辨率训练",
                        "优化anchor设计"
                    ],
                    'confidence_issues': [
                        "使用置信度校准技术",
                        "调整分类损失权重",
                        "使用标签平滑",
                        "增加困难样本挖掘"
                    ]
                }
                
                print(f"\n{error_names.get(error_type, error_type)} ({count}个):")
                for suggestion in suggestions.get(error_type, []):
                    print(f"  • {suggestion}")
    
    print(f"\n📁 详细结果位置:")
    print(f"  分析目录: {analysis_dir}")
    print(f"  错误图像: {analysis_dir}/[split]/[error_type]/")
    print(f"  统计文件: {analysis_dir}/overall_statistics.json")
    
    print("\n" + "=" * 60)
    print("分析完成! 可以运行 error_analysis.ipynb 进行详细可视化分析")

if __name__ == "__main__":
    import sys
    
    analysis_dir = sys.argv[1] if len(sys.argv) > 1 else 'test_prediction_analysis'
    view_analysis_results(analysis_dir)
