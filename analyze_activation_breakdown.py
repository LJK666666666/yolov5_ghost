#!/usr/bin/env python3
"""
分析激活参数比例差异的详细脚本.

解释为什么实际激活比例比MoE层理论激活比例大很多

Author: Augment Agent (Claude Sonnet 4 by Anthropic)
Created: 2025-07-05
Description: 深入分析YOLOv5 MoE模型中激活参数比例与理论MoE层激活比例差异的原因
"""

from pathlib import Path

import yaml


def analyze_parameter_breakdown(yaml_path: str, model_name: str):
    """详细分析参数构成."""
    print(f"\n🔍 {model_name} 参数构成分析")
    print("=" * 60)

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 统计不同类型的层
    backbone = config.get("backbone", [])
    head = config.get("head", [])
    all_layers = backbone + head

    layer_stats = {"Conv": 0, "C3": 0, "C3MoE": 0, "MoEConv": 0, "AdaptiveMoE": 0, "SPPF": 0, "Other": 0}

    moe_layers = []
    standard_layers = []

    for i, layer_config in enumerate(all_layers):
        if len(layer_config) < 4:
            continue

        from_layer, number, module, args = layer_config[:4]

        if module in layer_stats:
            layer_stats[module] += 1
        else:
            layer_stats["Other"] += 1

        # 分类MoE和标准层
        if module in ["C3MoE", "MoEConv", "AdaptiveMoE"]:
            moe_layers.append({"index": i, "module": module, "args": args})
        else:
            standard_layers.append({"index": i, "module": module, "args": args})

    print("📊 层类型统计:")
    total_layers = sum(layer_stats.values())
    for layer_type, count in layer_stats.items():
        if count > 0:
            percentage = count / total_layers * 100
            print(f"   {layer_type}: {count}层 ({percentage:.1f}%)")

    print("\n🎯 MoE vs 标准层:")
    moe_count = len(moe_layers)
    standard_count = len(standard_layers)
    print(f"   MoE层: {moe_count}层 ({moe_count / (moe_count + standard_count) * 100:.1f}%)")
    print(f"   标准层: {standard_count}层 ({standard_count / (moe_count + standard_count) * 100:.1f}%)")

    # 分析MoE层的激活比例
    if moe_layers:
        print("\n🔬 MoE层详细分析:")
        total_experts = 0
        total_activated = 0

        for layer in moe_layers:
            module = layer["module"]
            args = layer["args"]

            if module == "C3MoE" and len(args) >= 6:
                num_experts = args[4]
                top_k = args[5]
                total_experts += num_experts
                total_activated += top_k
                print(f"   {module}: {num_experts}专家 → 激活{top_k}个 ({top_k / num_experts:.1%})")

            elif module == "MoEConv" and len(args) >= 8:
                num_experts = args[6]
                top_k = args[7]
                total_experts += num_experts
                total_activated += top_k
                print(f"   {module}: {num_experts}专家 → 激活{top_k}个 ({top_k / num_experts:.1%})")

            elif module == "AdaptiveMoE" and len(args) >= 9:
                max_experts = args[6]
                min_top_k = args[7]
                max_top_k = args[8]
                avg_top_k = (min_top_k + max_top_k) / 2
                total_experts += max_experts
                total_activated += avg_top_k
                print(
                    f"   {module}: {max_experts}专家 → 激活{min_top_k}-{max_top_k}个 (平均{avg_top_k / max_experts:.1%})"
                )

        if total_experts > 0:
            moe_activation_ratio = total_activated / total_experts
            print(f"\n📈 MoE层平均激活比例: {moe_activation_ratio:.1%}")

    return {
        "total_layers": total_layers,
        "moe_layers": moe_count,
        "standard_layers": standard_count,
        "moe_percentage": moe_count / (moe_count + standard_count) * 100 if (moe_count + standard_count) > 0 else 0,
    }


def explain_activation_ratio_difference():
    """解释激活比例差异的原因."""
    print("\n" + "=" * 80)
    print("🤔 为什么实际激活比例比MoE层激活比例大这么多？")
    print("=" * 80)

    print("""
🎯 主要原因分析:

1. **非MoE层的影响**
   • 标准Conv层: 100%激活
   • 标准C3层: 100%激活  
   • SPPF层: 100%激活
   • Head中的Detect层: 100%激活
   
2. **MoE层占比较小**
   • 轻量级MoE: 只有6个MoE层，大部分仍是标准层
   • 完整MoE: 虽有16个MoE层，但仍有很多标准层
   • 自适应MoE: 只有7个MoE层
   
3. **Head部分的影响**
   • Head部分参数量很大(包含Detect层)
   • 大部分Head层仍是标准层，100%激活
   • 即使Head中有MoE层，占比也不高

4. **参数分布不均**
   • 深层网络的参数量更大
   • 如果深层使用标准层，会大幅提高整体激活比例
   
🔍 具体计算公式:
整体激活比例 = (标准层参数×100% + MoE层参数×MoE激活比例) / 总参数

📊 举例说明:
假设模型有:
- 标准层参数: 15M (100%激活)
- MoE层参数: 5M (30%激活 = 1.5M)
- 总激活参数: 15M + 1.5M = 16.5M
- 总参数: 15M + 5M = 20M  
- 整体激活比例: 16.5M/20M = 82.5%

虽然MoE层只有30%激活，但由于标准层占大头，整体激活比例仍然很高！
""")


def main():
    """主函数."""
    models = {
        "YOLOv5s-MoE-Lite": "models/yolov5s-moe-lite.yaml",
        "YOLOv5s-MoE": "models/yolov5s-moe.yaml",
        "YOLOv5s-Adaptive-MoE": "models/yolov5s-adaptive-moe.yaml",
    }

    results = {}

    for model_name, yaml_path in models.items():
        if Path(yaml_path).exists():
            result = analyze_parameter_breakdown(yaml_path, model_name)
            results[model_name] = result

    # 总结分析
    print("\n" + "=" * 80)
    print("📋 MoE层占比总结")
    print("=" * 80)

    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"   MoE层占比: {result['moe_percentage']:.1f}%")
        print(f"   标准层数量: {result['standard_layers']}层")
        print(f"   MoE层数量: {result['moe_layers']}层")
        print()

    explain_activation_ratio_difference()


if __name__ == "__main__":
    main()
