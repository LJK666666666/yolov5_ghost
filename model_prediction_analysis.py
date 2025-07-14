#!/usr/bin/env python3
"""
YOLOv5模型预测分析脚本
对SafetyVests.v6数据集进行预测并分析预测错误.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import torch
import yaml

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

                    annotations.append(
                        {
                            "class_id": class_id,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": 1.0,  # 真实标签置信度为1
                        }
                    )

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


def match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5):
    """将预测结果与真实标签进行匹配."""
    matched_predictions = []
    matched_gt = []
    unmatched_predictions = []
    unmatched_gt = list(ground_truth)

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(unmatched_gt):
            if pred["class_id"] == gt["class_id"]:
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = i

        if best_gt_idx >= 0:
            matched_predictions.append({"prediction": pred, "ground_truth": unmatched_gt[best_gt_idx], "iou": best_iou})
            matched_gt.append(unmatched_gt.pop(best_gt_idx))
        else:
            unmatched_predictions.append(pred)

    return matched_predictions, unmatched_predictions, unmatched_gt


def analyze_prediction_errors_by_class(matched, unmatched_pred, unmatched_gt, class_names):
    """按类别分析预测错误."""
    # 初始化每个类别的统计
    class_stats = {}
    for i, class_name in enumerate(class_names):
        class_stats[i] = {
            "class_name": class_name,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "low_iou_matches": 0,
            "confidence_issues": 0,
            "tp_details": [],
            "fp_details": [],
            "fn_details": [],
            "low_iou_details": [],
            "confidence_details": [],
        }

    # 统计True Positives (正确匹配)
    for match in matched:
        pred_class = match["prediction"]["class_id"]
        gt_class = match["ground_truth"]["class_id"]

        if pred_class == gt_class:  # 类别匹配
            if match["iou"] >= 0.7:  # IoU足够高
                class_stats[pred_class]["true_positives"] += 1
                class_stats[pred_class]["tp_details"].append(
                    {
                        "type": "true_positive",
                        "class": class_names[pred_class],
                        "confidence": match["prediction"]["confidence"],
                        "iou": match["iou"],
                        "pred_bbox": match["prediction"]["bbox"],
                        "true_bbox": match["ground_truth"]["bbox"],
                    }
                )
            else:  # IoU较低
                class_stats[pred_class]["low_iou_matches"] += 1
                class_stats[pred_class]["low_iou_details"].append(
                    {
                        "type": "low_iou",
                        "predicted_class": class_names[pred_class],
                        "true_class": class_names[gt_class],
                        "confidence": match["prediction"]["confidence"],
                        "iou": match["iou"],
                        "pred_bbox": match["prediction"]["bbox"],
                        "true_bbox": match["ground_truth"]["bbox"],
                    }
                )

            # 检查置信度问题
            if match["prediction"]["confidence"] < 0.5:
                class_stats[pred_class]["confidence_issues"] += 1
                class_stats[pred_class]["confidence_details"].append(
                    {
                        "type": "low_confidence",
                        "predicted_class": class_names[pred_class],
                        "true_class": class_names[gt_class],
                        "confidence": match["prediction"]["confidence"],
                        "iou": match["iou"],
                    }
                )
        else:  # 类别不匹配 - 这种情况在当前匹配逻辑中不应该发生
            # 如果发生，记录为该预测类别的FP和真实类别的FN
            class_stats[pred_class]["false_positives"] += 1
            class_stats[pred_class]["fp_details"].append(
                {
                    "type": "class_mismatch_fp",
                    "predicted_class": class_names[pred_class],
                    "true_class": class_names[gt_class],
                    "confidence": match["prediction"]["confidence"],
                    "bbox": match["prediction"]["bbox"],
                }
            )

            class_stats[gt_class]["false_negatives"] += 1
            class_stats[gt_class]["fn_details"].append(
                {
                    "type": "class_mismatch_fn",
                    "true_class": class_names[gt_class],
                    "predicted_class": class_names[pred_class],
                    "bbox": match["ground_truth"]["bbox"],
                }
            )

    # 统计False Positives (未匹配的预测)
    for pred in unmatched_pred:
        pred_class = pred["class_id"]
        class_stats[pred_class]["false_positives"] += 1
        class_stats[pred_class]["fp_details"].append(
            {
                "type": "false_positive",
                "predicted_class": class_names[pred_class],
                "confidence": pred["confidence"],
                "bbox": pred["bbox"],
            }
        )

    # 统计False Negatives (未匹配的真实标签)
    for gt in unmatched_gt:
        gt_class = gt["class_id"]
        class_stats[gt_class]["false_negatives"] += 1
        class_stats[gt_class]["fn_details"].append(
            {"type": "false_negative", "true_class": class_names[gt_class], "bbox": gt["bbox"]}
        )

    return class_stats


def draw_predictions_and_ground_truth(image, predictions, ground_truth, class_names):
    """在图像上绘制预测结果和真实标签."""
    img_copy = image.copy()

    # 绘制真实标签 (绿色)
    for gt in ground_truth:
        x1, y1, x2, y2 = gt["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_copy, f"GT: {class_names[gt['class_id']]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # 绘制预测结果 (红色)
    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img_copy,
            f"Pred: {class_names[pred['class_id']]} ({pred['confidence']:.2f})",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    return img_copy


def process_dataset_split(model, images_path, labels_path, output_dir, split_name, class_names, conf_threshold=0.25):
    """处理数据集的一个分割."""
    print(f"\n处理{split_name}...")

    # 创建输出目录
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    # 创建子目录
    error_dirs = {
        "false_positives": os.path.join(split_output_dir, "false_positives"),
        "false_negatives": os.path.join(split_output_dir, "false_negatives"),
        "low_iou_matches": os.path.join(split_output_dir, "low_iou_matches"),
        "confidence_issues": os.path.join(split_output_dir, "confidence_issues"),
        "correct_predictions": os.path.join(split_output_dir, "correct_predictions"),
    }

    for dir_path in error_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图像文件
    image_files = glob.glob(os.path.join(images_path, "*.jpg"))

    all_results = []
    error_stats = defaultdict(int)
    correct_count = 0

    for i, img_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(image_files)}")

        # 读取图像
        image = cv2.imread(img_file)
        if image is None:
            continue

        img_height, img_width = image.shape[:2]

        # 获取对应的标签文件
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(labels_path, f"{img_name}.txt")

        # 解析真实标签
        ground_truth = parse_yolo_label(label_file, img_width, img_height)

        # 模型预测
        results = model(img_file, size=640)
        predictions = []

        # 解析预测结果
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= conf_threshold:
                predictions.append({"class_id": int(cls), "bbox": [int(x) for x in box], "confidence": float(conf)})

        # 匹配预测结果和真实标签
        matched, unmatched_pred, unmatched_gt = match_predictions_to_ground_truth(predictions, ground_truth)

        # 按类别分析错误
        class_stats = analyze_prediction_errors_by_class(matched, unmatched_pred, unmatched_gt, class_names)

        # 统计错误类型（为了兼容性保留旧的统计方式）
        has_errors = False
        total_fp = sum(stats["false_positives"] for stats in class_stats.values())
        total_fn = sum(stats["false_negatives"] for stats in class_stats.values())
        total_low_iou = sum(stats["low_iou_matches"] for stats in class_stats.values())
        total_conf_issues = sum(stats["confidence_issues"] for stats in class_stats.values())

        if total_fp > 0:
            has_errors = True
            error_stats["false_positives"] += total_fp
        if total_fn > 0:
            has_errors = True
            error_stats["false_negatives"] += total_fn
        if total_low_iou > 0:
            has_errors = True
            error_stats["low_iou_matches"] += total_low_iou
        if total_conf_issues > 0:
            has_errors = True
            error_stats["confidence_issues"] += total_conf_issues

        # 保存结果
        result_data = {
            "image_file": img_file,
            "image_name": img_name,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "matched": matched,
            "class_stats": class_stats,
            "has_errors": has_errors,
        }
        all_results.append(result_data)

        # 绘制并保存有错误的图像
        if has_errors:
            annotated_img = draw_predictions_and_ground_truth(image, predictions, ground_truth, class_names)

            # 保存按类别统计的错误图像
            if total_fp > 0:
                save_path = os.path.join(error_dirs["false_positives"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

                # 收集所有FP错误详情
                fp_errors = []
                for stats in class_stats.values():
                    fp_errors.extend(stats["fp_details"])

                error_info = {
                    "image_name": img_name,
                    "errors": fp_errors,
                    "total_predictions": len(predictions),
                    "total_ground_truth": len(ground_truth),
                    "class_breakdown": {
                        class_names[i]: stats["false_positives"]
                        for i, stats in class_stats.items()
                        if stats["false_positives"] > 0
                    },
                }

                info_path = os.path.join(error_dirs["false_positives"], f"{img_name}_info.json")
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(error_info, f, indent=2, ensure_ascii=False)

            if total_fn > 0:
                save_path = os.path.join(error_dirs["false_negatives"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

                # 收集所有FN错误详情
                fn_errors = []
                for stats in class_stats.values():
                    fn_errors.extend(stats["fn_details"])

                error_info = {
                    "image_name": img_name,
                    "errors": fn_errors,
                    "total_predictions": len(predictions),
                    "total_ground_truth": len(ground_truth),
                    "class_breakdown": {
                        class_names[i]: stats["false_negatives"]
                        for i, stats in class_stats.items()
                        if stats["false_negatives"] > 0
                    },
                }

                info_path = os.path.join(error_dirs["false_negatives"], f"{img_name}_info.json")
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(error_info, f, indent=2, ensure_ascii=False)

            if total_low_iou > 0:
                save_path = os.path.join(error_dirs["low_iou_matches"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

                # 收集所有低IoU错误详情
                low_iou_errors = []
                for stats in class_stats.values():
                    low_iou_errors.extend(stats["low_iou_details"])

                error_info = {
                    "image_name": img_name,
                    "errors": low_iou_errors,
                    "total_predictions": len(predictions),
                    "total_ground_truth": len(ground_truth),
                    "class_breakdown": {
                        class_names[i]: stats["low_iou_matches"]
                        for i, stats in class_stats.items()
                        if stats["low_iou_matches"] > 0
                    },
                }

                info_path = os.path.join(error_dirs["low_iou_matches"], f"{img_name}_info.json")
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(error_info, f, indent=2, ensure_ascii=False)

            if total_conf_issues > 0:
                save_path = os.path.join(error_dirs["confidence_issues"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

                # 收集所有置信度问题详情
                conf_errors = []
                for stats in class_stats.values():
                    conf_errors.extend(stats["confidence_details"])

                error_info = {
                    "image_name": img_name,
                    "errors": conf_errors,
                    "total_predictions": len(predictions),
                    "total_ground_truth": len(ground_truth),
                    "class_breakdown": {
                        class_names[i]: stats["confidence_issues"]
                        for i, stats in class_stats.items()
                        if stats["confidence_issues"] > 0
                    },
                }

                info_path = os.path.join(error_dirs["confidence_issues"], f"{img_name}_info.json")
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(error_info, f, indent=2, ensure_ascii=False)
        else:
            correct_count += 1
            # 保存少量正确预测的示例
            if correct_count <= 20:  # 只保存前20个正确的示例
                annotated_img = draw_predictions_and_ground_truth(image, predictions, ground_truth, class_names)
                save_path = os.path.join(error_dirs["correct_predictions"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

    # 计算按类别的总体统计
    overall_class_stats = {}
    for i, class_name in enumerate(class_names):
        total_tp = sum(result["class_stats"][i]["true_positives"] for result in all_results if "class_stats" in result)
        total_fp = sum(result["class_stats"][i]["false_positives"] for result in all_results if "class_stats" in result)
        total_fn = sum(result["class_stats"][i]["false_negatives"] for result in all_results if "class_stats" in result)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        overall_class_stats[class_name] = {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    # 保存统计信息
    stats = {
        "total_images": len(image_files),
        "correct_predictions": correct_count,
        "error_statistics": dict(error_stats),
        "accuracy": correct_count / len(image_files) if image_files else 0,
        "class_statistics": overall_class_stats,
    }

    stats_path = os.path.join(split_output_dir, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 保存详细结果
    results_path = os.path.join(split_output_dir, "detailed_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        # 简化结果以减少文件大小
        simplified_results = []
        for result in all_results:
            # 统计该图像的错误类型
            error_types = []
            if "class_stats" in result:
                for class_stat in result["class_stats"].values():
                    if class_stat["false_positives"] > 0:
                        error_types.append("false_positives")
                    if class_stat["false_negatives"] > 0:
                        error_types.append("false_negatives")
                    if class_stat["low_iou_matches"] > 0:
                        error_types.append("low_iou_matches")
                    if class_stat["confidence_issues"] > 0:
                        error_types.append("confidence_issues")

            simplified_results.append(
                {
                    "image_name": result["image_name"],
                    "has_errors": result["has_errors"],
                    "num_predictions": len(result["predictions"]),
                    "num_ground_truth": len(result["ground_truth"]),
                    "error_types": list(set(error_types)),  # 去重
                }
            )
        json.dump(simplified_results, f, indent=2, ensure_ascii=False)

    print(f"{split_name}处理完成:")
    print(f"  总图像数: {len(image_files)}")
    print(f"  正确预测: {correct_count}")
    print(f"  准确率: {correct_count / len(image_files) * 100:.2f}%")
    print(f"  错误统计: {dict(error_stats)}")

    return stats, all_results


def main():
    parser = argparse.ArgumentParser(description="YOLOv5模型预测分析")
    parser.add_argument("--weights", default="runs/train200to300epoch/yolov5s_/weights/best.pt", help="模型权重路径")
    parser.add_argument("--data", default="data/SafetyVests.v6", help="数据集路径")
    parser.add_argument("--output", default="prediction_analysis", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.1, help="置信度阈值")

    args = parser.parse_args()

    print("YOLOv5模型预测分析")
    print("=" * 50)

    # 检查模型文件
    if not os.path.exists(args.weights):
        print(f"错误: 模型文件不存在 {args.weights}")
        return

    # 检查数据集
    if not os.path.exists(args.data):
        print(f"错误: 数据集路径不存在 {args.data}")
        return

    # 加载模型
    print(f"加载模型: {args.weights}")
    model = load_model(args.weights)
    if model is None:
        return

    # 读取数据集配置
    config_file = os.path.join(args.data, "data.yaml")
    with open(config_file) as f:
        data_config = yaml.safe_load(f)

    class_names = data_config["names"]
    print(f"类别: {class_names}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 处理各个数据集分割
    splits = {
        "train": ("train/images", "train/labels"),
        "valid": ("valid/images", "valid/labels"),
        "test": ("test/images", "test/labels"),
    }

    all_stats = {}

    for split_name, (img_dir, label_dir) in splits.items():
        images_path = os.path.join(args.data, img_dir)
        labels_path = os.path.join(args.data, label_dir)

        if os.path.exists(images_path) and os.path.exists(labels_path):
            stats, _ = process_dataset_split(
                model, images_path, labels_path, args.output, split_name, class_names, args.conf
            )
            all_stats[split_name] = stats
        else:
            print(f"跳过{split_name}: 路径不存在")

    # 保存总体统计
    overall_stats_path = os.path.join(args.output, "overall_statistics.json")
    with open(overall_stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n分析完成! 结果保存在: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
