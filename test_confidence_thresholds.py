#!/usr/bin/env python3
"""测试不同置信度阈值对模型性能的影响."""

import glob
import json
import os
import sys

import torch

# 添加YOLOv5路径
sys.path.append(".")


def load_model(weights_path):
    """加载YOLOv5模型."""
    try:
        model = torch.hub.load(".", "custom", path=weights_path, source="local", force_reload=True)
        model.eval()
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def parse_yolo_label(label_file, img_width, img_height):
    """解析YOLO格式标签文件."""
    annotations = []
    if os.path.exists(label_file):
        with open(label_file) as f:
            lines = f.readlines()

        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    annotations.append({"class_id": class_id, "bbox": [x1, y1, x2, y2], "confidence": 1.0})

    return annotations


def calculate_iou(box1, box2):
    """计算两个边界框的IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_at_threshold(model, images_path, labels_path, conf_threshold, iou_threshold=0.5):
    """在特定置信度阈值下评估模型."""
    image_files = glob.glob(os.path.join(images_path, "*.jpg"))

    total_predictions = 0
    total_ground_truth = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for img_file in image_files[:100]:  # 只测试前100张图像以加快速度
        # 读取图像尺寸
        import cv2

        image = cv2.imread(img_file)
        if image is None:
            continue
        img_height, img_width = image.shape[:2]

        # 获取真实标签
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(labels_path, f"{img_name}.txt")
        ground_truth = parse_yolo_label(label_file, img_width, img_height)

        # 模型预测
        results = model(img_file, size=640)
        predictions = []

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= conf_threshold:
                predictions.append({"class_id": int(cls), "bbox": [int(x) for x in box], "confidence": float(conf)})

        total_predictions += len(predictions)
        total_ground_truth += len(ground_truth)

        # 匹配预测和真实标签
        matched_gt = []
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(ground_truth):
                if i not in matched_gt and pred["class_id"] == gt["class_id"]:
                    iou = calculate_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = i

            if best_gt_idx >= 0:
                true_positives += 1
                matched_gt.append(best_gt_idx)
            else:
                false_positives += 1

        # 未匹配的真实标签为漏检
        false_negatives += len(ground_truth) - len(matched_gt)

    # 计算指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "conf_threshold": conf_threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predictions": total_predictions,
        "total_ground_truth": total_ground_truth,
    }


def main():
    """主函数."""
    print("测试不同置信度阈值对模型性能的影响")
    print("=" * 60)

    # 配置
    weights_path = "runs/train200to300epoch/yolov5s_/weights/best.pt"
    data_path = "data/SafetyVests.v6"

    # 加载模型
    print(f"加载模型: {weights_path}")
    model = load_model(weights_path)
    if model is None:
        return

    # 测试不同置信度阈值
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

    # 使用验证集进行测试
    images_path = os.path.join(data_path, "valid/images")
    labels_path = os.path.join(data_path, "valid/labels")

    results = []

    print("\n在验证集上测试不同置信度阈值...")
    print("-" * 60)
    print(f"{'阈值':<6} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
    print("-" * 60)

    for threshold in thresholds:
        result = evaluate_at_threshold(model, images_path, labels_path, threshold)
        results.append(result)

        print(
            f"{threshold:<6.2f} {result['precision']:<8.3f} {result['recall']:<8.3f} "
            f"{result['f1_score']:<8.3f} {result['true_positives']:<4} "
            f"{result['false_positives']:<4} {result['false_negatives']:<4}"
        )

    # 找到最佳F1分数的阈值
    best_result = max(results, key=lambda x: x["f1_score"])

    print("-" * 60)
    print(f"最佳置信度阈值: {best_result['conf_threshold']:.2f}")
    print(f"最佳F1分数: {best_result['f1_score']:.3f}")
    print(f"对应精确率: {best_result['precision']:.3f}")
    print(f"对应召回率: {best_result['recall']:.3f}")

    # 保存结果
    output_file = "confidence_threshold_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

    # 给出建议
    print("\n💡 建议:")
    if best_result["conf_threshold"] < 0.25:
        print(f"  • 当前默认阈值(0.25)可能过高，建议降低到 {best_result['conf_threshold']:.2f}")
        print("  • 这将提高召回率，减少漏检")
    elif best_result["conf_threshold"] > 0.25:
        print(f"  • 当前默认阈值(0.25)可能过低，建议提高到 {best_result['conf_threshold']:.2f}")
        print("  • 这将提高精确率，减少误检")
    else:
        print("  • 当前默认阈值(0.25)已经是最佳选择")

    print("\n使用建议的阈值重新运行分析:")
    print(f"python model_prediction_analysis.py --conf {best_result['conf_threshold']:.2f}")


if __name__ == "__main__":
    main()
