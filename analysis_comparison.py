#!/usr/bin/env python3
"""
分析方法对比脚本
比较传统分析方法和层次化分析方法的结果差异
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_traditional_stats(traditional_dir):
    """加载传统分析结果"""
    stats_file = Path(traditional_dir) / 'overall_statistics.json'
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_hierarchical_stats(hierarchical_dir):
    """加载层次化分析结果"""
    stats_file = Path(hierarchical_dir) / 'overall_hierarchical_statistics.json'
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_analysis_methods(traditional_stats, hierarchical_stats):
    """比较两种分析方法"""
    
    print("=" * 80)
    print("传统分析 vs 层次化分析对比")
    print("=" * 80)
    
    for split in ['train', 'valid', 'test']:
        if split in traditional_stats and split in hierarchical_stats:
            print(f"\n{split.upper()} 数据集:")
            print("-" * 40)
            
            trad = traditional_stats[split]
            hier = hierarchical_stats[split]
            
            # 基本准确率对比
            print(f"整体准确率:")
            print(f"  传统方法: {trad['accuracy']*100:.2f}%")
            print(f"  层次化方法: {hier['accuracy']*100:.2f}%")
            print(f"  差异: {(hier['accuracy'] - trad['accuracy'])*100:.2f}%")
            
            # Person检测性能对比
            if 'person_detection' in hier:
                print(f"\nPerson检测性能 (层次化方法):")
                print(f"  Precision: {hier['person_detection']['precision']:.3f}")
                print(f"  Recall: {hier['person_detection']['recall']:.3f}")
                print(f"  F1-Score: {hier['person_detection']['f1_score']:.3f}")
            
            # 传统方法的类别统计
            if 'class_statistics' in trad:
                print(f"\n传统方法各类别性能:")
                for class_name, stats in trad['class_statistics'].items():
                    print(f"  {class_name}:")
                    print(f"    Precision: {stats['precision']:.3f}")
                    print(f"    Recall: {stats['recall']:.3f}")
                    print(f"    F1-Score: {stats['f1_score']:.3f}")
            
            # 装备状态分析 (仅层次化方法)
            if 'equipment_accuracy' in hier:
                print(f"\n装备状态准确率 (层次化方法):")
                for status, accuracy in hier['equipment_accuracy'].items():
                    print(f"  {status}: {accuracy:.3f}")
            
            # 错误统计对比
            print(f"\n错误统计对比:")
            if 'error_statistics' in trad:
                print(f"  传统方法错误:")
                for error_type, count in trad['error_statistics'].items():
                    print(f"    {error_type}: {count}")
            
            if 'person_detection' in hier:
                print(f"  层次化方法Person检测错误:")
                print(f"    false_positives: {hier['person_detection']['false_positives']}")
                print(f"    false_negatives: {hier['person_detection']['false_negatives']}")

def create_comparison_visualization(traditional_stats, hierarchical_stats, output_dir):
    """创建对比可视化图表"""
    
    # 准备数据
    splits = ['train', 'valid', 'test']
    traditional_acc = []
    hierarchical_acc = []
    
    for split in splits:
        if split in traditional_stats and split in hierarchical_stats:
            traditional_acc.append(traditional_stats[split]['accuracy'] * 100)
            hierarchical_acc.append(hierarchical_stats[split]['accuracy'] * 100)
        else:
            traditional_acc.append(0)
            hierarchical_acc.append(0)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 整体准确率对比
    ax1 = axes[0, 0]
    x = range(len(splits))
    width = 0.35
    ax1.bar([i - width/2 for i in x], traditional_acc, width, label='传统方法', alpha=0.8)
    ax1.bar([i + width/2 for i in x], hierarchical_acc, width, label='层次化方法', alpha=0.8)
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('整体准确率对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Person检测性能 (仅层次化方法)
    ax2 = axes[0, 1]
    person_metrics = ['precision', 'recall', 'f1_score']
    train_person = [hierarchical_stats['train']['person_detection'][m] for m in person_metrics]
    valid_person = [hierarchical_stats['valid']['person_detection'][m] for m in person_metrics]
    test_person = [hierarchical_stats['test']['person_detection'][m] for m in person_metrics]
    
    x = range(len(person_metrics))
    ax2.plot(x, train_person, 'o-', label='Train', linewidth=2, markersize=8)
    ax2.plot(x, valid_person, 's-', label='Valid', linewidth=2, markersize=8)
    ax2.plot(x, test_person, '^-', label='Test', linewidth=2, markersize=8)
    ax2.set_xlabel('指标')
    ax2.set_ylabel('分数')
    ax2.set_title('Person检测性能 (层次化方法)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. 装备状态准确率
    ax3 = axes[1, 0]
    equipment_types = ['fully_equipped', 'helmet_only', 'vest_only', 'no_equipment']
    train_equipment = [hierarchical_stats['train']['equipment_accuracy'][eq] for eq in equipment_types]
    valid_equipment = [hierarchical_stats['valid']['equipment_accuracy'][eq] for eq in equipment_types]
    test_equipment = [hierarchical_stats['test']['equipment_accuracy'][eq] for eq in equipment_types]
    
    x = range(len(equipment_types))
    width = 0.25
    ax3.bar([i - width for i in x], train_equipment, width, label='Train', alpha=0.8)
    ax3.bar(x, valid_equipment, width, label='Valid', alpha=0.8)
    ax3.bar([i + width for i in x], test_equipment, width, label='Test', alpha=0.8)
    ax3.set_xlabel('装备状态')
    ax3.set_ylabel('准确率')
    ax3.set_title('装备状态分类准确率')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['完全装备', '仅头盔', '仅背心', '无装备'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 错误数量对比
    ax4 = axes[1, 1]
    traditional_errors = []
    hierarchical_errors = []
    
    for split in splits:
        if split in traditional_stats:
            trad_total = sum(traditional_stats[split]['error_statistics'].values())
            traditional_errors.append(trad_total)
        else:
            traditional_errors.append(0)
            
        if split in hierarchical_stats:
            hier_total = (hierarchical_stats[split]['person_detection']['false_positives'] + 
                         hierarchical_stats[split]['person_detection']['false_negatives'])
            hierarchical_errors.append(hier_total)
        else:
            hierarchical_errors.append(0)
    
    x = range(len(splits))
    ax4.bar([i - width/2 for i in x], traditional_errors, width, label='传统方法', alpha=0.8)
    ax4.bar([i + width/2 for i in x], hierarchical_errors, width, label='层次化方法', alpha=0.8)
    ax4.set_xlabel('数据集')
    ax4.set_ylabel('错误数量')
    ax4.set_title('检测错误数量对比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(splits)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n对比图表已保存到: {output_dir}/analysis_comparison.png")

def main():
    traditional_dir = 'traditional_analysis'
    hierarchical_dir = 'hierarchical_analysis'
    output_dir = 'comparison_results'
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 加载数据
    print("加载分析结果...")
    traditional_stats = load_traditional_stats(traditional_dir)
    hierarchical_stats = load_hierarchical_stats(hierarchical_dir)
    
    # 进行对比分析
    compare_analysis_methods(traditional_stats, hierarchical_stats)
    
    # 创建可视化
    print("\n创建对比可视化...")
    create_comparison_visualization(traditional_stats, hierarchical_stats, output_dir)
    
    # 保存对比报告
    comparison_report = {
        'summary': {
            'traditional_method': '传统独立类别分析方法',
            'hierarchical_method': '层次化安全装备分析方法',
            'key_differences': [
                '层次化方法考虑了person与装备的关联关系',
                '避免了将person身上的装备误判为独立错误',
                '提供了装备穿戴状态的详细分析',
                '更准确地反映了安全装备检测的实际性能'
            ]
        },
        'traditional_stats': traditional_stats,
        'hierarchical_stats': hierarchical_stats
    }
    
    with open(f'{output_dir}/comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整对比报告已保存到: {output_dir}/comparison_report.json")

if __name__ == "__main__":
    main()
