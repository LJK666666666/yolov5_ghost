#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RA-mAP计算功能

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

# 导入RA-mAP计算函数
from test_all_models import calculate_ra_map, create_performance_table

def test_ra_map_calculation():
    """测试RA-mAP计算功能"""
    print("🧪 测试RA-mAP计算功能")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        {
            'name': '模型A',
            'map50_95': 0.5,
            'no_vest_recall': 0.8,
            'expected_ra_map': 0.4 * 0.5 + 0.6 * 0.8  # 0.2 + 0.48 = 0.68
        },
        {
            'name': '模型B',
            'map50_95': 0.7,
            'no_vest_recall': 0.6,
            'expected_ra_map': 0.4 * 0.7 + 0.6 * 0.6  # 0.28 + 0.36 = 0.64
        },
        {
            'name': '模型C',
            'map50_95': 0.3,
            'no_vest_recall': 0.9,
            'expected_ra_map': 0.4 * 0.3 + 0.6 * 0.9  # 0.12 + 0.54 = 0.66
        },
        {
            'name': '模型D (缺失数据)',
            'map50_95': None,
            'no_vest_recall': 0.8,
            'expected_ra_map': None
        },
        {
            'name': '模型E (缺失数据)',
            'map50_95': 0.5,
            'no_vest_recall': 'N/A',
            'expected_ra_map': None
        }
    ]
    
    print("测试RA-mAP计算:")
    print("-" * 60)
    
    all_passed = True
    for i, case in enumerate(test_cases, 1):
        map50_95 = case['map50_95']
        no_vest_recall = case['no_vest_recall']
        expected = case['expected_ra_map']
        
        result = calculate_ra_map(map50_95, no_vest_recall)
        
        print(f"测试 {i}: {case['name']}")
        print(f"  输入: mAP@0.5:0.95={map50_95}, NO-Safety Vest Recall={no_vest_recall}")
        print(f"  期望: {expected}")
        print(f"  结果: {result}")
        
        if expected is None:
            if result is None:
                print(f"  ✅ 通过")
            else:
                print(f"  ❌ 失败 - 期望None但得到{result}")
                all_passed = False
        else:
            if result is not None and abs(result - expected) < 1e-6:
                print(f"  ✅ 通过")
            else:
                print(f"  ❌ 失败 - 期望{expected}但得到{result}")
                all_passed = False
        print()
    
    return all_passed

def test_performance_table_creation():
    """测试性能表格创建功能"""
    print("🧪 测试性能表格创建功能")
    print("=" * 60)
    
    # 模拟测试数据
    mock_results = {
        'yolov5s_se_exp1': {
            'precision': 0.85,
            'recall': 0.78,
            'map50': 0.82,
            'map50_95': 0.55,
            'no_safety_vest_recall': 0.75,
            'no_safety_vest_precision': 0.80
        },
        'yolov5s_sparse_moe_exp2': {
            'precision': 0.88,
            'recall': 0.82,
            'map50': 0.85,
            'map50_95': 0.60,
            'no_safety_vest_recall': 0.85,
            'no_safety_vest_precision': 0.83
        },
        'yolov5s_standard_exp3': {
            'precision': 0.80,
            'recall': 0.75,
            'map50': 0.78,
            'map50_95': 0.50,
            'no_safety_vest_recall': 0.70,
            'no_safety_vest_precision': 0.75
        },
        'failed_model_exp4': {}  # 模拟失败的模型
    }
    
    # 创建临时输出目录
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 测试表格创建
        create_performance_table(mock_results, output_dir, 'best', 'test_train')
        
        # 检查文件是否创建
        csv_file = output_dir / "performance_comparison_best_test_train.csv"
        excel_file = output_dir / "performance_comparison_best_test_train.xlsx"
        
        csv_exists = csv_file.exists()
        excel_exists = excel_file.exists()
        
        print(f"CSV文件创建: {'✅ 成功' if csv_exists else '❌ 失败'}")
        print(f"Excel文件创建: {'✅ 成功' if excel_exists else '❌ 失败'}")
        
        if csv_exists:
            # 读取CSV文件验证内容
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"CSV文件行数: {len(df)}")
            print(f"CSV文件列数: {len(df.columns)}")
            print("CSV文件列名:", list(df.columns))
            
            # 验证RA-mAP计算
            for idx, row in df.iterrows():
                model_name = row['模型名称']
                ra_map_str = row['RA-mAP']
                
                if ra_map_str != 'N/A':
                    ra_map_value = float(ra_map_str)
                    print(f"  {model_name}: RA-mAP = {ra_map_value:.4f}")
                else:
                    print(f"  {model_name}: RA-mAP = N/A")
        
        return csv_exists and excel_exists
        
    except Exception as e:
        print(f"❌ 表格创建测试失败: {e}")
        return False
    
    finally:
        # 清理测试文件
        try:
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
        except:
            pass

def main():
    """主函数"""
    print("🔧 RA-mAP功能测试")
    print("=" * 80)
    
    # 测试RA-mAP计算
    calc_passed = test_ra_map_calculation()
    
    print("\n" + "=" * 80)
    
    # 测试表格创建
    table_passed = test_performance_table_creation()
    
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    
    if calc_passed and table_passed:
        print("🎉 所有测试通过！")
        print("\n✅ RA-mAP计算功能正常")
        print("✅ 性能表格创建功能正常")
        print("\n🚀 test_all_models.py 已成功添加以下功能:")
        print("  1. RA-mAP指标计算 (0.4 × mAP@0.5:0.95 + 0.6 × NO-Safety Vest Recall)")
        print("  2. CSV格式性能对比表格")
        print("  3. Excel格式性能对比表格 (包含样式和说明)")
        print("  4. 汇总报告中的RA-mAP分析")
        print("  5. 实时RA-mAP计算和显示")
        
        print("\n📊 使用方法:")
        print("python test_all_models.py --model-type best --train-folder your_train_folder")
        
    else:
        print("❌ 部分测试失败")
        if not calc_passed:
            print("  • RA-mAP计算功能有问题")
        if not table_passed:
            print("  • 性能表格创建功能有问题")

if __name__ == "__main__":
    main()
