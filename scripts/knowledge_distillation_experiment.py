#!/usr/bin/env python3
"""
知识蒸馏对比实验脚本
创新亮点四：引入"知识蒸馏"提升小模型性能

使用方法：
python scripts/knowledge_distillation_experiment.py --mode [baseline|distill|compare|test|all]
"""

import os
import sys
import argparse
import subprocess
import torch
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def test_teacher_model():
    """测试教师模型是否可用"""
    print("🔍 测试教师模型...")
    
    teacher_weights = ROOT / "runs/sv6_train1000epoch_/yolov5x_/weights/best.pt"
    
    if not teacher_weights.exists():
        print(f"❌ 教师模型权重文件不存在: {teacher_weights}")
        return False
    
    try:
        # 测试加载教师模型
        from models.yolo import Model

        ckpt = torch.load(teacher_weights, map_location='cpu')

        # 获取教师模型的原始类别数
        teacher_nc = ckpt["model"].yaml.get('nc', 3)
        print(f"📚 教师模型类别数: {teacher_nc}")

        # 使用教师模型的原始配置
        teacher_model = Model(ckpt["model"].yaml, ch=3, nc=teacher_nc)
        teacher_csd = ckpt["model"].float().state_dict()
        teacher_model.load_state_dict(teacher_csd, strict=False)

        # 计算参数量
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"✅ 教师模型加载成功: {teacher_params:,} 参数")

        # 测试学生模型（使用数据集配置的类别数）
        import yaml
        with open(ROOT / "data/SafetyVests.v6/data.yaml", 'r') as f:
            data_config = yaml.safe_load(f)
        student_nc = data_config.get('nc', 3)
        print(f"🎓 学生模型类别数: {student_nc}")

        student_model = Model(ROOT / "models/yolov5s.yaml", ch=3, nc=student_nc)
        student_params = sum(p.numel() for p in student_model.parameters())
        print(f"✅ 学生模型加载成功: {student_params:,} 参数")

        print(f"📊 参数比例: 学生/教师 = {student_params/teacher_params:.2%}")

        # 检查类别数兼容性
        if teacher_nc != student_nc:
            print(f"⚠️  警告: 教师模型({teacher_nc}类)与学生模型({student_nc}类)类别数不匹配")
            print("   知识蒸馏可能需要调整，但仍可进行实验")

        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def run_baseline_training():
    """运行基线训练（无知识蒸馏）"""
    cmd = [
        "python", "train.py",
        "--cfg", "models/yolov5s.yaml",
        "--data", "data/SafetyVests.v6/data.yaml",
        "--weights", "yolov5s.pt",
        "--project", "runs/examination",
        "--name", "baseline_no_distill",
        "--epochs", "1",
        "--batch-size", "32"
    ]
    
    print("🚀 开始基线训练（无知识蒸馏）")
    print(f"📝 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=True)
        print("✅ 基线训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 基线训练失败: {e}")
        return False

def run_distillation_training():
    """运行知识蒸馏训练"""
    teacher_weights = ROOT / "runs/sv6_train1000epoch_/yolov5x_/weights/best.pt"
    
    cmd = [
        "python", "train.py",
        "--cfg", "models/yolov5s.yaml",
        "--data", "data/SafetyVests.v6/data.yaml",
        "--weights", "yolov5s.pt",
        "--project", "runs/examination",
        "--name", "distill_yolov5s",
        "--epochs", "1",
        "--batch-size", "32",
        "--distillation",
        "--teacher-weights", str(teacher_weights),
        "--distill-alpha", "0.7",
        "--distill-temp", "4.0"
    ]
    
    print("🎓 开始知识蒸馏训练")
    print(f"📝 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=True)
        print("✅ 知识蒸馏训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 知识蒸馏训练失败: {e}")
        return False

def run_validation(weights_path, experiment_name):
    """运行验证实验"""
    cmd = [
        "python", "val.py",
        "--data", "data/SafetyVests.v6/data.yaml",
        "--weights", str(weights_path),
        "--name", f"{experiment_name}_val",
        "--save-txt", "--save-conf"
    ]
    
    print(f"🔍 开始验证实验: {experiment_name}")
    print(f"📝 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"✅ 验证完成: {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 验证失败: {experiment_name}, 错误: {e}")
        return False

def compare_results():
    """对比实验结果"""
    print("\n" + "="*60)
    print("📊 知识蒸馏实验结果对比")
    print("="*60)
    
    experiments = [
        ("baseline_no_distill", "基线模型（无蒸馏）"),
        ("distill_yolov5s", "知识蒸馏模型"),
    ]
    
    for exp_name, exp_desc in experiments:
        results_file = ROOT / f"runs/examination/{exp_name}/results.csv"
        weights_file = ROOT / f"runs/examination/{exp_name}/weights/best.pt"
        
        if results_file.exists():
            print(f"✅ {exp_desc}: {results_file}")
        else:
            print(f"❌ {exp_desc}: 结果文件不存在")
            
        if weights_file.exists():
            print(f"   权重文件: {weights_file}")
        else:
            print(f"   权重文件: 不存在")
    
    print("\n💡 请查看以下目录的详细结果:")
    for exp_name, exp_desc in experiments:
        print(f"   - {exp_desc}: runs/examination/{exp_name}/")
    print(f"   - 验证结果: runs/val/")
    
    # 教师模型信息
    teacher_weights = ROOT / "runs/sv6_train1000epoch_/yolov5x_/weights/best.pt"
    if teacher_weights.exists():
        print(f"\n🎓 教师模型: {teacher_weights}")
    else:
        print(f"\n❌ 教师模型不存在: {teacher_weights}")

def main():
    parser = argparse.ArgumentParser(description="知识蒸馏对比实验")
    parser.add_argument("--mode", choices=["baseline", "distill", "compare", "test", "all"], 
                       default="test", help="实验模式")
    
    args = parser.parse_args()
    
    print("🎓 引入'知识蒸馏'提升小模型性能实验")
    print("="*60)
    
    if args.mode in ["test", "all"]:
        print("\n📋 步骤0: 测试教师模型")
        if not test_teacher_model():
            print("❌ 教师模型测试失败，请检查模型文件")
            return
    
    if args.mode in ["baseline", "all"]:
        print("\n📋 步骤1: 基线实验（无知识蒸馏）")
        success = run_baseline_training()
        
        if success:
            weights_path = ROOT / "runs/examination/baseline_no_distill/weights/best.pt"
            run_validation(weights_path, "baseline_no_distill")
    
    if args.mode in ["distill", "all"]:
        print("\n📋 步骤2: 知识蒸馏实验")
        success = run_distillation_training()
        
        if success:
            weights_path = ROOT / "runs/examination/distill_yolov5s/weights/best.pt"
            run_validation(weights_path, "distill_yolov5s")
    
    if args.mode in ["compare", "all"]:
        print("\n📋 步骤3: 结果对比")
        compare_results()
    
    print("\n🎉 实验完成！")
    print("\n📝 项目故事素材:")
    print('"在模型压缩的需求下，我们希望用轻量级的YOLOv5s达到接近大模型YOLOv5x的性能。')
    print('为此，我们引入了知识蒸馏技术，让训练好的YOLOv5x作为"教师"，指导YOLOv5s')
    print('这个"学生"的学习过程。通过让学生模型不仅学习真实标签，还学习教师模型的')
    print('"软标签"（概率分布），学生模型能够获得更丰富的知识，从而在保持轻量化的')
    print('同时显著提升检测性能。实验结果表明，经过知识蒸馏的YOLOv5s相比基线版本')
    print('在mAP上提升了X%，接近了大模型的性能水平。"')

if __name__ == "__main__":
    main()
