#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析runs/train200epoch下所有best.pt模型的参数量、计算速度和模型大小
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# 尝试导入thop库用于计算GFLOPs
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    print("警告: thop库未安装，无法计算GFLOPs。请运行: pip install thop")
    THOP_AVAILABLE = False

def get_model_size_mb(model_path):
    """获取模型文件大小（MB）"""
    try:
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except Exception as e:
        print(f"获取模型大小失败 {model_path}: {e}")
        return 0

def count_parameters(model):
    """计算模型参数量"""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    except Exception as e:
        print(f"计算参数量失败: {e}")
        return 0, 0

def calculate_gflops(model, device, input_size=(640, 640)):
    """计算模型的GFLOPs"""
    if not THOP_AVAILABLE:
        return 0
    
    try:
        model.eval()
        model.to(device)
        
        # 创建测试输入
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # 使用thop计算GFLOPs
        with torch.no_grad():
            result = profile(model, inputs=(dummy_input,), verbose=False)
            if isinstance(result, tuple) and len(result) >= 2:
                flops, params = result[0], result[1]
            else:
                print("thop返回格式异常")
                return 0
        
        # 确保flops是标量
        if isinstance(flops, torch.Tensor):
            flops = flops.item()
        
        gflops = flops / (1024 ** 3)  # 转换为GFLOPs
        return gflops
        
    except Exception as e:
        print(f"计算GFLOPs失败: {e}")
        return 0

def measure_inference_speed(model, device, input_size=(640, 640), num_runs=100):
    """测量推理速度"""
    try:
        model.eval()
        model.to(device)
        
        # 创建测试输入
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测量推理时间
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / num_runs * 1000  # 转换为毫秒
        fps = num_runs / total_time
        
        return avg_time_per_image, fps
        
    except Exception as e:
        print(f"测量推理速度失败: {e}")
        return 0, 0

def analyze_model(model_path, device):
    """分析单个模型"""
    model_info = {
        'model_path': str(model_path),
        'model_name': model_path.parent.parent.name,
        'file_size_mb': 0,
        'total_params': 0,
        'trainable_params': 0,
        'gflops': 0,
        'inference_time_ms': 0,
        'fps': 0,
        'error': None
    }
    
    try:
        # 获取文件大小
        model_info['file_size_mb'] = get_model_size_mb(model_path)
        
        # 加载模型
        model = DetectMultiBackend(model_path, device=device)
        
        # 计算参数量
        total_params, trainable_params = count_parameters(model.model)
        model_info['total_params'] = total_params
        model_info['trainable_params'] = trainable_params
        
        # 计算GFLOPs
        gflops = calculate_gflops(model.model, device)
        model_info['gflops'] = gflops
        
        # 测量推理速度
        inference_time, fps = measure_inference_speed(model.model, device)
        model_info['inference_time_ms'] = inference_time
        model_info['fps'] = fps
        
        print(f"✓ {model_info['model_name']}: {total_params:,} 参数, {gflops:.1f} GFLOPs, {inference_time:.2f}ms, {fps:.1f} FPS")
        
    except Exception as e:
        error_msg = str(e)
        model_info['error'] = error_msg
        print(f"✗ {model_info['model_name']}: 分析失败 - {error_msg}")
    
    return model_info

def get_all_best_models():
    """获取所有best.pt模型路径"""
    train_dir = Path("runs/train200epoch")
    models = []
    
    if not train_dir.exists():
        print(f"训练目录不存在: {train_dir}")
        return models
    
    for exp_dir in train_dir.iterdir():
        if exp_dir.is_dir():
            weights_dir = exp_dir / "weights"
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                models.append(best_pt)
    
    return models

def create_summary_report(models_info, output_path):
    """创建汇总报告"""
    # 创建DataFrame
    df_data = []
    for info in models_info:
        if info['error'] is None:
            df_data.append({
                '模型名称': info['model_name'],
                '文件大小(MB)': round(info['file_size_mb'], 2),
                '总参数量': f"{info['total_params']:,}",
                '可训练参数': f"{info['trainable_params']:,}",
                'GFLOPs': round(info['gflops'], 1),
                '推理时间(ms)': round(info['inference_time_ms'], 2),
                'FPS': round(info['fps'], 1),
                '状态': '成功'
            })
        else:
            df_data.append({
                '模型名称': info['model_name'],
                '文件大小(MB)': round(info['file_size_mb'], 2),
                '总参数量': 'N/A',
                '可训练参数': 'N/A',
                'GFLOPs': 'N/A',
                '推理时间(ms)': 'N/A',
                'FPS': 'N/A',
                '状态': f'失败: {info["error"]}'
            })
    
    df = pd.DataFrame(df_data)
    
    # 保存为Excel文件
    excel_path = output_path / "模型分析汇总.xlsx"
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    # 保存为CSV文件
    csv_path = output_path / "模型分析汇总.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 创建详细报告
    report_path = output_path / "模型分析详细报告.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("YOLOv5 模型分析报告\n")
        f.write("=" * 120 + "\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析模型数量: {len(models_info)}\n")
        f.write(f"成功分析: {len([m for m in models_info if m['error'] is None])}\n")
        f.write(f"分析失败: {len([m for m in models_info if m['error'] is not None])}\n")
        if not THOP_AVAILABLE:
            f.write("注意: thop库未安装，GFLOPs计算不可用\n")
        f.write("\n")
        
        # 成功分析的模型统计
        successful_models = [m for m in models_info if m['error'] is None]
        if successful_models:
            f.write("成功分析的模型统计:\n")
            f.write("-" * 60 + "\n")
            
            # 参数量统计
            total_params = [m['total_params'] for m in successful_models]
            f.write(f"参数量范围: {min(total_params):,} - {max(total_params):,}\n")
            f.write(f"平均参数量: {np.mean(total_params):,.0f}\n")
            
            # GFLOPs统计
            if THOP_AVAILABLE:
                gflops_values = [m['gflops'] for m in successful_models]
                f.write(f"GFLOPs范围: {min(gflops_values):.1f} - {max(gflops_values):.1f}\n")
                f.write(f"平均GFLOPs: {np.mean(gflops_values):.1f}\n")
            
            # 文件大小统计
            file_sizes = [m['file_size_mb'] for m in successful_models]
            f.write(f"文件大小范围: {min(file_sizes):.2f}MB - {max(file_sizes):.2f}MB\n")
            f.write(f"平均文件大小: {np.mean(file_sizes):.2f}MB\n")
            
            # 推理速度统计
            inference_times = [m['inference_time_ms'] for m in successful_models]
            f.write(f"推理时间范围: {min(inference_times):.2f}ms - {max(inference_times):.2f}ms\n")
            f.write(f"平均推理时间: {np.mean(inference_times):.2f}ms\n")
            
            fps_values = [m['fps'] for m in successful_models]
            f.write(f"FPS范围: {min(fps_values):.1f} - {max(fps_values):.1f}\n")
            f.write(f"平均FPS: {np.mean(fps_values):.1f}\n\n")
        
        # 详细结果表格
        f.write("详细分析结果:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'模型名称':<25} {'文件大小(MB)':<12} {'参数量':<15} {'GFLOPs':<8} {'推理时间(ms)':<12} {'FPS':<8} {'状态':<20}\n")
        f.write("-" * 120 + "\n")
        
        for info in models_info:
            if info['error'] is None:
                gflops_str = f"{info['gflops']:.1f}" if THOP_AVAILABLE else "N/A"
                f.write(f"{info['model_name']:<25} {info['file_size_mb']:<12.2f} {info['total_params']:<15,} {gflops_str:<8} {info['inference_time_ms']:<12.2f} {info['fps']:<8.1f} {'成功':<20}\n")
            else:
                error_msg = f"失败: {info['error']}"
                f.write(f"{info['model_name']:<25} {info['file_size_mb']:<12.2f} {'N/A':<15} {'N/A':<8} {'N/A':<12} {'N/A':<8} {error_msg:<20}\n")
        
        f.write("-" * 120 + "\n")
        
        # 找出最佳模型
        if successful_models:
            best_fps_model = max(successful_models, key=lambda x: x['fps'])
            f.write(f"\n最快推理模型: {best_fps_model['model_name']} ({best_fps_model['fps']:.1f} FPS)\n")
            
            # 找出最小模型（基于文件大小）
            smallest_model = min(successful_models, key=lambda x: x['file_size_mb'])
            f.write(f"最小模型: {smallest_model['model_name']} ({smallest_model['file_size_mb']:.2f} MB)\n")
            
            # 找出参数量最少的模型
            smallest_params_model = min(successful_models, key=lambda x: x['total_params'])
            f.write(f"参数量最少模型: {smallest_params_model['model_name']} ({smallest_params_model['total_params']:,} 参数)\n")
            
            # 找出GFLOPs最少的模型
            if THOP_AVAILABLE:
                smallest_gflops_model = min(successful_models, key=lambda x: x['gflops'])
                f.write(f"计算量最少模型: {smallest_gflops_model['model_name']} ({smallest_gflops_model['gflops']:.1f} GFLOPs)\n")
    
    print(f"汇总报告已保存到: {output_path}")
    print(f"Excel文件: {excel_path}")
    print(f"CSV文件: {csv_path}")
    print(f"详细报告: {report_path}")

def main():
    """主函数"""
    print("开始分析所有best.pt模型...")
    
    # 检查thop库是否可用
    if not THOP_AVAILABLE:
        print("警告: thop库未安装，GFLOPs计算将不可用")
        print("请运行: pip install thop")
    
    # 获取所有模型
    models = get_all_best_models()
    if not models:
        print("未找到任何best.pt模型文件！")
        return
    
    print(f"找到 {len(models)} 个模型:")
    for model_path in models:
        print(f"  - {model_path.parent.parent.name}: {model_path}")
    
    # 选择设备
    device = select_device('')
    print(f"\n使用设备: {device}")
    
    # 分析每个模型
    models_info = []
    for i, model_path in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 分析模型: {model_path.parent.parent.name}")
        model_info = analyze_model(model_path, device)
        models_info.append(model_info)
    
    # 创建输出目录
    output_path = Path("runs/train200epoch")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建汇总报告
    create_summary_report(models_info, output_path)
    
    # 保存详细JSON结果
    json_path = output_path / "模型分析详细结果.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(models_info, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n分析完成！")
    print(f"详细JSON结果: {json_path}")

if __name__ == "__main__":
    main() 