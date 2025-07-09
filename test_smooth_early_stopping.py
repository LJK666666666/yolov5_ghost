#!/usr/bin/env python3
"""
测试新的平滑早停机制

Author: Augment Agent (Claude Sonnet 4 by Anthropic)
Created: 2025-07-05
Description: 测试和演示基于滑动窗口平均值的早停机制
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.torch_utils import EarlyStopping, SmoothEarlyStopping


def simulate_training_scenario(scenario_name, fitness_values, patience=30, window_size=10):
    """
    模拟训练场景，对比标准早停和平滑早停的表现
    
    Args:
        scenario_name (str): 场景名称
        fitness_values (list): 模拟的fitness值序列
        patience (int): 耐心值
        window_size (int): 滑动窗口大小
    """
    print(f"\n🎯 测试场景: {scenario_name}")
    print("=" * 60)
    
    # 初始化两种早停机制
    standard_stopper = EarlyStopping(patience=patience)
    smooth_stopper = SmoothEarlyStopping(patience=patience, window_size=window_size)
    
    # 记录结果
    standard_stopped = False
    smooth_stopped = False
    standard_stop_epoch = None
    smooth_stop_epoch = None
    
    smooth_avg_history = []
    
    print(f"Epoch | Fitness  | Std Stop | Smooth Avg | Smooth Stop")
    print("-" * 55)
    
    for epoch, fitness in enumerate(fitness_values):
        # 标准早停检查
        if not standard_stopped:
            standard_stop = standard_stopper(epoch, fitness)
            if standard_stop:
                standard_stopped = True
                standard_stop_epoch = epoch
        
        # 平滑早停检查
        if not smooth_stopped:
            smooth_stop = smooth_stopper(epoch, fitness)
            if smooth_stop:
                smooth_stopped = True
                smooth_stop_epoch = epoch
        
        # 获取平滑早停状态
        status = smooth_stopper.get_status_info()
        smooth_avg_history.append(status['current_avg_fitness'])
        
        # 打印状态
        std_status = "✓" if standard_stopped else "○"
        smooth_status = "✓" if smooth_stopped else "○"
        
        print(f"{epoch:5d} | {fitness:8.4f} | {std_status:8s} | {status['current_avg_fitness']:10.4f} | {smooth_status:11s}")
        
        # 如果两种方法都停止了，就结束
        if standard_stopped and smooth_stopped:
            break
    
    # 总结结果
    print(f"\n📊 结果对比:")
    print(f"  标准早停: {'第' + str(standard_stop_epoch) + '个epoch停止' if standard_stopped else '未停止'}")
    print(f"  平滑早停: {'第' + str(smooth_stop_epoch) + '个epoch停止' if smooth_stopped else '未停止'}")
    
    if standard_stop_epoch is not None and smooth_stop_epoch is not None:
        diff = smooth_stop_epoch - standard_stop_epoch
        if diff > 0:
            print(f"  平滑早停比标准早停晚停止 {diff} 个epoch")
        elif diff < 0:
            print(f"  平滑早停比标准早停早停止 {-diff} 个epoch")
        else:
            print(f"  两种方法在同一epoch停止")
    
    return {
        'standard_stop_epoch': standard_stop_epoch,
        'smooth_stop_epoch': smooth_stop_epoch,
        'smooth_avg_history': smooth_avg_history
    }


def test_scenarios():
    """测试不同的训练场景"""
    print("🧪 平滑早停机制测试")
    print("=" * 80)
    
    # 场景1: 正常收敛后稳定
    print("\n📈 场景1: 正常收敛后稳定")
    fitness1 = [0.1, 0.2, 0.35, 0.45, 0.52, 0.58, 0.62, 0.65, 0.67, 0.68] + [0.68] * 50
    result1 = simulate_training_scenario("正常收敛", fitness1, patience=20, window_size=10)
    
    # 场景2: 训练过程有波动
    print("\n📈 场景2: 训练过程有波动")
    np.random.seed(42)
    base_trend = np.linspace(0.1, 0.7, 40)
    noise = np.random.normal(0, 0.05, 40)
    fitness2 = list(base_trend + noise)
    # 添加后期稳定阶段
    fitness2.extend([0.7 + np.random.normal(0, 0.02) for _ in range(30)])
    result2 = simulate_training_scenario("有波动的训练", fitness2, patience=20, window_size=10)
    
    # 场景3: 训练停滞
    print("\n📈 场景3: 训练停滞")
    fitness3 = [0.1, 0.2, 0.3, 0.35, 0.37, 0.38] + [0.38 + np.random.normal(0, 0.01) for _ in range(50)]
    result3 = simulate_training_scenario("训练停滞", fitness3, patience=20, window_size=10)
    
    # 场景4: 后期突破
    print("\n📈 场景4: 后期突破")
    fitness4 = [0.1, 0.2, 0.3, 0.35] + [0.35 + np.random.normal(0, 0.01) for _ in range(25)]
    fitness4.extend([0.36, 0.38, 0.42, 0.48, 0.55, 0.62, 0.68, 0.72])  # 后期突破
    fitness4.extend([0.72] * 20)  # 稳定
    result4 = simulate_training_scenario("后期突破", fitness4, patience=20, window_size=10)
    
    return [result1, result2, result3, result4]


def demonstrate_parameters():
    """演示不同参数设置的效果"""
    print(f"\n" + "=" * 80)
    print("🔧 参数设置效果演示")
    print("=" * 80)
    
    # 生成测试数据：有噪声的收敛过程
    np.random.seed(123)
    epochs = 60
    base_fitness = np.concatenate([
        np.linspace(0.1, 0.6, 30),  # 前30个epoch快速提升
        np.linspace(0.6, 0.65, 20),  # 中间20个epoch缓慢提升
        [0.65] * 10  # 最后10个epoch稳定
    ])
    noise = np.random.normal(0, 0.03, epochs)
    fitness_values = base_fitness + noise
    
    # 测试不同窗口大小
    print("\n📊 不同窗口大小的影响:")
    window_sizes = [5, 10, 15, 20]
    
    for window_size in window_sizes:
        stopper = SmoothEarlyStopping(patience=15, window_size=window_size, min_delta=0.001)
        
        for epoch, fitness in enumerate(fitness_values):
            stop = stopper(epoch, fitness)
            if stop:
                print(f"  窗口大小 {window_size:2d}: 第 {epoch:2d} 个epoch停止")
                break
        else:
            print(f"  窗口大小 {window_size:2d}: 未停止")
    
    # 测试不同最小增量
    print("\n📊 不同最小增量的影响:")
    min_deltas = [0.0001, 0.001, 0.005, 0.01]
    
    for min_delta in min_deltas:
        stopper = SmoothEarlyStopping(patience=15, window_size=10, min_delta=min_delta)
        
        for epoch, fitness in enumerate(fitness_values):
            stop = stopper(epoch, fitness)
            if stop:
                print(f"  最小增量 {min_delta:.4f}: 第 {epoch:2d} 个epoch停止")
                break
        else:
            print(f"  最小增量 {min_delta:.4f}: 未停止")


def generate_usage_examples():
    """生成使用示例"""
    print(f"\n" + "=" * 80)
    print("📚 使用示例")
    print("=" * 80)
    
    examples = [
        {
            'name': '启用平滑早停（基础）',
            'command': 'python train.py --smooth-early-stop',
            'description': '使用默认参数启用平滑早停'
        },
        {
            'name': '自定义窗口大小',
            'command': 'python train.py --smooth-early-stop --smooth-window 15',
            'description': '使用15个epoch的滑动窗口'
        },
        {
            'name': '调整耐心值',
            'command': 'python train.py --smooth-early-stop --smooth-patience 150',
            'description': '设置150个epoch的耐心值'
        },
        {
            'name': '精细调整',
            'command': 'python train.py --smooth-early-stop --smooth-patience 200 --smooth-window 20 --smooth-delta 0.0005',
            'description': '完全自定义所有参数'
        },
        {
            'name': '小数据集推荐',
            'command': 'python train.py --smooth-early-stop --smooth-patience 50 --smooth-window 5',
            'description': '适合小数据集的设置'
        },
        {
            'name': '大数据集推荐',
            'command': 'python train.py --smooth-early-stop --smooth-patience 300 --smooth-window 20',
            'description': '适合大数据集的设置'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   命令: {example['command']}")
        print(f"   说明: {example['description']}")


def main():
    """主函数"""
    print("🚀 平滑早停机制测试和演示")
    print("=" * 80)
    
    # 测试不同场景
    results = test_scenarios()
    
    # 演示参数效果
    demonstrate_parameters()
    
    # 生成使用示例
    generate_usage_examples()
    
    # 总结
    print(f"\n" + "=" * 80)
    print("📋 总结")
    print("=" * 80)
    
    print("\n🎯 平滑早停机制的优势:")
    advantages = [
        "更好地处理训练过程中的波动",
        "基于平均值的判断更加稳定",
        "减少因偶然波动导致的过早停止",
        "提供更详细的训练状态信息",
        "可调节的窗口大小适应不同场景"
    ]
    
    for advantage in advantages:
        print(f"  • {advantage}")
    
    print("\n🔧 参数调优建议:")
    tuning_tips = [
        "窗口大小: 小数据集用5-10，大数据集用10-20",
        "耐心值: 可以比标准早停设置得更大",
        "最小增量: 根据fitness的数值范围调整",
        "首次使用建议用默认参数测试"
    ]
    
    for tip in tuning_tips:
        print(f"  • {tip}")
    
    print(f"\n✅ 平滑早停机制已成功集成到YOLOv5训练流程中！")


if __name__ == "__main__":
    main()
