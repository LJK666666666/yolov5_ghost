#!/usr/bin/env python3
"""
YOLOv5 模型激活参数计算脚本

计算不同YOLOv5模型变体的激活参数数量：
- yolov5s.yaml: 标准模型，所有参数都激活
- yolov5s-moe-lite.yaml: 轻量级MoE，部分专家激活
- yolov5s-moe.yaml: 完整MoE，部分专家激活  
- yolov5s-adaptive-moe.yaml: 自适应MoE，动态专家激活

激活参数 = 实际前向传播时使用的参数数量
对于MoE模型，只有被选中的专家参数才算激活参数
"""

import yaml
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any


class ParameterCalculator:
    """参数计算器"""
    
    def __init__(self):
        self.activation_params = {}
        self.total_params = {}
        
    def calculate_conv_params(self, c1: int, c2: int, k: int = 3, groups: int = 1) -> int:
        """计算卷积层参数数量"""
        # 卷积核参数 + 偏置参数 + BatchNorm参数
        conv_params = (c1 * c2 * k * k) // groups
        bn_params = c2 * 2  # gamma + beta
        return conv_params + bn_params
    
    def calculate_linear_params(self, c1: int, c2: int) -> int:
        """计算线性层参数数量"""
        return c1 * c2 + c2  # 权重 + 偏置
    
    def calculate_c3_params(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> int:
        """计算C3模块参数数量"""
        c_ = int(c2 * e)  # hidden channels
        
        # cv1: Conv(c1, c_, 1, 1)
        cv1_params = self.calculate_conv_params(c1, c_, 1)
        
        # cv2: Conv(c1, c_, 1, 1) 
        cv2_params = self.calculate_conv_params(c1, c_, 1)
        
        # cv3: Conv(2 * c_, c2, 1)
        cv3_params = self.calculate_conv_params(2 * c_, c2, 1)
        
        # m: n个Bottleneck
        bottleneck_params = 0
        for _ in range(n):
            # Bottleneck: cv1(c_, c_//2, 1) + cv2(c_//2, c_, 3)
            bottleneck_params += self.calculate_conv_params(c_, c_//2, 1)
            bottleneck_params += self.calculate_conv_params(c_//2, c_, 3)
            
        return cv1_params + cv2_params + cv3_params + bottleneck_params
    
    def calculate_sppf_params(self, c1: int, c2: int, k: int = 5) -> int:
        """计算SPPF模块参数数量"""
        c_ = c1 // 2
        cv1_params = self.calculate_conv_params(c1, c_, 1)
        cv2_params = self.calculate_conv_params(c_ * 4, c2, 1)
        return cv1_params + cv2_params
    
    def calculate_expert_params(self, c1: int, c2: int, k: int = 3, expert_type: str = 'conv') -> int:
        """计算单个专家的参数数量"""
        if expert_type == 'conv':
            return self.calculate_conv_params(c1, c2, k)
        elif expert_type == 'bottleneck':
            c_ = c2 // 4
            params = self.calculate_conv_params(c1, c_, 1)  # 降维
            params += self.calculate_conv_params(c_, c_, k)  # 主卷积
            params += self.calculate_conv_params(c_, c2, 1)  # 升维
            return params
        elif expert_type == 'dwconv':
            params = self.calculate_conv_params(c1, c1, k, groups=c1)  # DW卷积
            params += self.calculate_conv_params(c1, c2, 1)  # PW卷积
            return params
        else:
            return self.calculate_conv_params(c1, c2, k)
    
    def calculate_moe_layer_params(self, c1: int, c2: int, num_experts: int, top_k: int, 
                                 k: int = 3, expert_type: str = 'conv') -> Tuple[int, int]:
        """计算MoE层的总参数和激活参数"""
        # 专家网络参数
        expert_params = self.calculate_expert_params(c1, c2, k, expert_type)
        total_expert_params = expert_params * num_experts
        
        # 门控网络参数
        gate_params = self.calculate_linear_params(c1, num_experts)
        
        # 总参数
        total_params = total_expert_params + gate_params
        
        # 激活参数 = 门控网络参数 + top_k个专家的参数
        activation_params = gate_params + expert_params * top_k
        
        return total_params, activation_params


class ModelAnalyzer:
    """模型分析器"""
    
    def __init__(self):
        self.calc = ParameterCalculator()
        
    def parse_yaml_config(self, yaml_path: str) -> Dict[str, Any]:
        """解析YAML配置文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def apply_multipliers(self, channels: int, config: Dict[str, Any]) -> int:
        """应用宽度倍数"""
        width_multiple = config.get('width_multiple', 1.0)
        return int(channels * width_multiple)
    
    def analyze_standard_yolov5s(self, config: Dict[str, Any]) -> Tuple[int, int]:
        """分析标准YOLOv5s模型"""
        total_params = 0
        
        # Backbone分析
        backbone = config['backbone']
        current_channels = 3  # 输入RGB图像
        
        for layer_config in backbone:
            from_layer, number, module, args = layer_config
            
            if module == 'Conv':
                c2, k, s = args[:3]
                c2 = self.apply_multipliers(c2, config)
                params = self.calc.calculate_conv_params(current_channels, c2, k)
                total_params += params
                current_channels = c2
                
            elif module == 'C3':
                c2 = args[0]
                c2 = self.apply_multipliers(c2, config)
                n = int(number * config.get('depth_multiple', 1.0))
                params = self.calc.calculate_c3_params(current_channels, c2, n)
                total_params += params
                current_channels = c2
                
            elif module == 'SPPF':
                c2, k = args[:2]
                c2 = self.apply_multipliers(c2, config)
                params = self.calc.calculate_sppf_params(current_channels, c2, k)
                total_params += params
                current_channels = c2
        
        # Head分析
        head = config['head']
        # 简化head分析，主要关注backbone的差异
        head_params = self.estimate_head_params(config)
        total_params += head_params
        
        # 对于标准模型，激活参数 = 总参数
        return total_params, total_params
    
    def analyze_moe_model(self, config: Dict[str, Any], model_type: str) -> Tuple[int, int]:
        """分析MoE模型"""
        total_params = 0
        activation_params = 0
        
        # Backbone分析
        backbone = config['backbone']
        current_channels = 3
        
        for layer_config in backbone:
            from_layer, number, module, args = layer_config
            
            if module == 'Conv':
                c2, k, s = args[:3]
                c2 = self.apply_multipliers(c2, config)
                params = self.calc.calculate_conv_params(current_channels, c2, k)
                total_params += params
                activation_params += params
                current_channels = c2
                
            elif module == 'C3':
                c2 = args[0]
                c2 = self.apply_multipliers(c2, config)
                n = int(number * config.get('depth_multiple', 1.0))
                params = self.calc.calculate_c3_params(current_channels, c2, n)
                total_params += params
                activation_params += params
                current_channels = c2
                
            elif module == 'C3MoE':
                c2, shortcut, g, e, num_experts, top_k = args[:6]
                c2 = self.apply_multipliers(c2, config)
                n = int(number * config.get('depth_multiple', 1.0))
                
                # C3MoE的基础结构参数
                c_ = int(c2 * e)
                cv1_params = self.calc.calculate_conv_params(current_channels, c_, 1)
                cv2_params = self.calc.calculate_conv_params(current_channels, c_, 1)
                cv3_params = self.calc.calculate_conv_params(2 * c_, c2, 1)
                base_params = cv1_params + cv2_params + cv3_params
                
                # MoE Bottleneck参数
                moe_total = 0
                moe_activation = 0
                for _ in range(n):
                    # cv1: 标准卷积
                    cv1_bottleneck = self.calc.calculate_conv_params(c_, c_, 1)
                    # cv2: MoE层
                    moe_t, moe_a = self.calc.calculate_moe_layer_params(c_, c2, num_experts, top_k, 3)
                    
                    moe_total += cv1_bottleneck + moe_t
                    moe_activation += cv1_bottleneck + moe_a
                
                total_params += base_params + moe_total
                activation_params += base_params + moe_activation
                current_channels = c2

            elif module == 'MoEConv':
                c2, k, s, p, g, act, num_experts, top_k = args[:8]
                c2 = self.apply_multipliers(c2, config)

                moe_t, moe_a = self.calc.calculate_moe_layer_params(
                    current_channels, c2, num_experts, top_k, k)
                total_params += moe_t
                activation_params += moe_a
                current_channels = c2

            elif module == 'AdaptiveMoE':
                c2, k, s, p, g, act, max_experts, min_top_k, max_top_k = args[:9]
                c2 = self.apply_multipliers(c2, config)

                # 自适应MoE：使用平均激活专家数
                avg_top_k = (min_top_k + max_top_k) / 2

                # 专家网络参数
                expert_params = self.calc.calculate_expert_params(current_channels, c2, k)
                total_expert_params = expert_params * max_experts

                # 门控网络和复杂度评估网络参数
                gate_params = self.calc.calculate_linear_params(current_channels, max_experts)
                complexity_params = self.calc.calculate_linear_params(current_channels, 1)

                total_params += total_expert_params + gate_params + complexity_params
                activation_params += expert_params * avg_top_k + gate_params + complexity_params
                current_channels = c2

            elif module == 'SPPF':
                c2, k = args[:2]
                c2 = self.apply_multipliers(c2, config)
                params = self.calc.calculate_sppf_params(current_channels, c2, k)
                total_params += params
                activation_params += params
                current_channels = c2

        # Head分析（简化）
        head_total, head_activation = self.estimate_moe_head_params(config, model_type)
        total_params += head_total
        activation_params += head_activation

        return total_params, activation_params

    def estimate_head_params(self, config: Dict[str, Any]) -> int:
        """估算标准head参数"""
        # 简化估算，基于典型的YOLOv5s head结构
        nc = config.get('nc', 80)

        # 典型head参数估算
        head_params = 0

        # 几个Conv层和Detect层
        head_params += self.calc.calculate_conv_params(1024, 512, 1)  # 降维
        head_params += self.calc.calculate_conv_params(512, 256, 1)   # 降维
        head_params += self.calc.calculate_conv_params(256, 256, 3)   # 上采样
        head_params += self.calc.calculate_conv_params(512, 512, 3)   # 上采样

        # Detect层参数估算
        na = 3  # anchors per output layer
        detect_params = (256 + 512 + 1024) * na * (nc + 5)  # 3个输出层
        head_params += detect_params

        return head_params

    def estimate_moe_head_params(self, config: Dict[str, Any], model_type: str) -> Tuple[int, int]:
        """估算MoE head参数"""
        base_head = self.estimate_head_params(config)

        if model_type == 'lite':
            # 轻量级MoE在head中有部分MoE层
            moe_ratio = 0.3  # 30%的head参数是MoE
            activation_ratio = 0.4  # MoE部分的激活比例
        elif model_type == 'full':
            # 完整MoE在head中大量使用MoE
            moe_ratio = 0.6  # 60%的head参数是MoE
            activation_ratio = 0.35  # MoE部分的激活比例
        elif model_type == 'adaptive':
            # 自适应MoE在head中适度使用MoE
            moe_ratio = 0.4  # 40%的head参数是MoE
            activation_ratio = 0.45  # MoE部分的激活比例
        else:
            return base_head, base_head

        moe_params = int(base_head * moe_ratio * 2)  # MoE增加参数
        standard_params = int(base_head * (1 - moe_ratio))

        total_head = standard_params + moe_params
        activation_head = standard_params + int(moe_params * activation_ratio)

        return total_head, activation_head


def analyze_activation_details(yaml_path: str, model_type: str) -> Dict[str, Any]:
    """分析模型的激活参数详情"""
    details = {
        'moe_layers': [],
        'activation_strategy': '',
        'expert_configs': []
    }

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if model_type == 'standard':
        details['activation_strategy'] = '全部参数激活'
        return details

    # 分析backbone和head中的MoE配置
    all_layers = config.get('backbone', []) + config.get('head', [])

    for i, layer_config in enumerate(all_layers):
        if len(layer_config) < 4:
            continue

        from_layer, number, module, args = layer_config[:4]

        if module == 'C3MoE':
            if len(args) >= 6:
                c2, shortcut, g, e, num_experts, top_k = args[:6]
                details['moe_layers'].append({
                    'layer_idx': i,
                    'module': module,
                    'num_experts': num_experts,
                    'top_k': top_k,
                    'activation_ratio': top_k / num_experts
                })

        elif module == 'MoEConv':
            if len(args) >= 8:
                c2, k, s, p, g, act, num_experts, top_k = args[:8]
                details['moe_layers'].append({
                    'layer_idx': i,
                    'module': module,
                    'num_experts': num_experts,
                    'top_k': top_k,
                    'activation_ratio': top_k / num_experts
                })

        elif module == 'AdaptiveMoE':
            if len(args) >= 9:
                c2, k, s, p, g, act, max_experts, min_top_k, max_top_k = args[:9]
                avg_top_k = (min_top_k + max_top_k) / 2
                details['moe_layers'].append({
                    'layer_idx': i,
                    'module': module,
                    'num_experts': max_experts,
                    'top_k_range': f"{min_top_k}-{max_top_k}",
                    'avg_top_k': avg_top_k,
                    'activation_ratio': avg_top_k / max_experts
                })

    # 设置激活策略描述
    if model_type == 'lite':
        details['activation_strategy'] = '选择性MoE激活'
    elif model_type == 'full':
        details['activation_strategy'] = '全面MoE激活'
    elif model_type == 'adaptive':
        details['activation_strategy'] = '自适应MoE激活'

    return details


def main():
    """主函数"""
    analyzer = ModelAnalyzer()

    models = {
        'YOLOv5s (标准)': ('models/yolov5s.yaml', 'standard'),
        'YOLOv5s-MoE-Lite (轻量级)': ('models/yolov5s-moe-lite.yaml', 'lite'),
        'YOLOv5s-MoE (完整)': ('models/yolov5s-moe.yaml', 'full'),
        'YOLOv5s-Adaptive-MoE (自适应)': ('models/yolov5s-adaptive-moe.yaml', 'adaptive')
    }

    print("=" * 80)
    print("YOLOv5 模型激活参数分析报告")
    print("=" * 80)
    print()

    results = {}

    for model_name, (yaml_path, model_type) in models.items():
        if not Path(yaml_path).exists():
            print(f"⚠️  文件不存在: {yaml_path}")
            continue

        try:
            config = analyzer.parse_yaml_config(yaml_path)

            if model_type == 'standard':
                total_params, activation_params = analyzer.analyze_standard_yolov5s(config)
            else:
                total_params, activation_params = analyzer.analyze_moe_model(config, model_type)

            results[model_name] = {
                'total_params': total_params,
                'activation_params': activation_params,
                'activation_ratio': activation_params / total_params if total_params > 0 else 0
            }

            # 获取激活详情
            details = analyze_activation_details(yaml_path, model_type)

            print(f"📊 {model_name}")
            print(f"   总参数量:     {total_params:,}")
            print(f"   激活参数量:   {activation_params:,}")
            print(f"   激活比例:     {activation_params/total_params*100:.1f}%")
            print(f"   参数效率:     {activation_params/1e6:.2f}M 激活参数")
            print(f"   激活策略:     {details['activation_strategy']}")

            # 显示MoE层详情
            if details['moe_layers']:
                print(f"   MoE层数量:    {len(details['moe_layers'])}个")
                for moe_layer in details['moe_layers'][:3]:  # 只显示前3个
                    if 'top_k_range' in moe_layer:
                        print(f"   └─ {moe_layer['module']}: {moe_layer['num_experts']}专家, "
                              f"激活{moe_layer['top_k_range']}个 "
                              f"(平均{moe_layer['activation_ratio']:.1%})")
                    else:
                        print(f"   └─ {moe_layer['module']}: {moe_layer['num_experts']}专家, "
                              f"激活{moe_layer['top_k']}个 "
                              f"({moe_layer['activation_ratio']:.1%})")
                if len(details['moe_layers']) > 3:
                    print(f"   └─ ... 还有{len(details['moe_layers'])-3}个MoE层")
            print()

        except Exception as e:
            print(f"❌ 分析 {model_name} 时出错: {e}")
            print()

    # 对比分析
    if len(results) > 1:
        print("=" * 80)
        print("模型对比分析")
        print("=" * 80)

        baseline = results.get('YOLOv5s (标准)', {})
        baseline_activation = baseline.get('activation_params', 1)

        for model_name, result in results.items():
            if model_name == 'YOLOv5s (标准)':
                continue

            activation_params = result['activation_params']
            ratio = activation_params / baseline_activation

            print(f"📈 {model_name}")
            print(f"   相对标准模型激活参数: {ratio:.2f}x")
            print(f"   激活参数增加: {(ratio-1)*100:+.1f}%")
            print()

    print("=" * 80)
    print("说明:")
    print("• 总参数量: 模型中所有参数的数量")
    print("• 激活参数量: 前向传播时实际使用的参数数量")
    print("• 对于标准模型: 激活参数 = 总参数")
    print("• 对于MoE模型: 激活参数 < 总参数 (只激活部分专家)")
    print("=" * 80)


if __name__ == "__main__":
    main()
