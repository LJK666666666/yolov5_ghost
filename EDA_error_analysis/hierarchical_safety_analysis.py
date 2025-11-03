#!/usr/bin/env python3
"""层次化安全装备检测分析脚本 专门处理person、helmet、safety_vest之间的层次关系 分析人员安全装备穿戴情况."""

import argparse
import glob
import json
import os
import sys

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


def calculate_overlap_ratio(small_box, large_box):
    """计算小框在大框中的重叠比例."""
    x1 = max(small_box[0], large_box[0])
    y1 = max(small_box[1], large_box[1])
    x2 = min(small_box[2], large_box[2])
    y2 = min(small_box[3], large_box[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])

    return intersection / small_area if small_area > 0 else 0.0


def analyze_person_equipment(person_bbox, helmets, safety_vests, overlap_threshold=0.3):
    """分析单个person的装备穿戴情况."""
    person_helmets = []
    person_vests = []

    # 检查helmet是否属于这个person
    for helmet in helmets:
        overlap_ratio = calculate_overlap_ratio(helmet["bbox"], person_bbox)
        if overlap_ratio >= overlap_threshold:
            person_helmets.append({**helmet, "overlap_ratio": overlap_ratio})

    # 检查safety_vest是否属于这个person
    for vest in safety_vests:
        overlap_ratio = calculate_overlap_ratio(vest["bbox"], person_bbox)
        if overlap_ratio >= overlap_threshold:
            person_vests.append({**vest, "overlap_ratio": overlap_ratio})

    # 确定person的装备状态
    has_helmet = len(person_helmets) > 0
    has_vest = len(person_vests) > 0

    if has_helmet and has_vest:
        equipment_status = "fully_equipped"
    elif has_helmet:
        equipment_status = "helmet_only"
    elif has_vest:
        equipment_status = "vest_only"
    else:
        equipment_status = "no_equipment"

    return {
        "equipment_status": equipment_status,
        "helmets": person_helmets,
        "vests": person_vests,
        "has_helmet": has_helmet,
        "has_vest": has_vest,
    }


def hierarchical_match_analysis(predictions, ground_truth, class_names, iou_threshold=0.5, overlap_threshold=0.3):
    """层次化匹配分析."""
    # 按类别分组
    pred_persons = [p for p in predictions if class_names[p["class_id"]] == "person"]
    pred_helmets = [p for p in predictions if class_names[p["class_id"]] == "helmet"]
    pred_vests = [p for p in predictions if class_names[p["class_id"]] == "safety_vest"]

    gt_persons = [g for g in ground_truth if class_names[g["class_id"]] == "person"]
    gt_helmets = [g for g in ground_truth if class_names[g["class_id"]] == "helmet"]
    gt_vests = [g for g in ground_truth if class_names[g["class_id"]] == "safety_vest"]

    # 匹配person
    matched_persons = []
    unmatched_pred_persons = []
    unmatched_gt_persons = list(gt_persons)

    for pred_person in pred_persons:
        best_iou = 0
        best_gt_idx = -1

        for i, gt_person in enumerate(unmatched_gt_persons):
            iou = calculate_iou(pred_person["bbox"], gt_person["bbox"])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = i

        if best_gt_idx >= 0:
            matched_gt = unmatched_gt_persons.pop(best_gt_idx)

            # 分析预测person的装备
            pred_equipment = analyze_person_equipment(pred_person["bbox"], pred_helmets, pred_vests, overlap_threshold)

            # 分析真实person的装备
            gt_equipment = analyze_person_equipment(matched_gt["bbox"], gt_helmets, gt_vests, overlap_threshold)

            matched_persons.append(
                {
                    "prediction": pred_person,
                    "ground_truth": matched_gt,
                    "iou": best_iou,
                    "pred_equipment": pred_equipment,
                    "gt_equipment": gt_equipment,
                }
            )
        else:
            # 分析未匹配预测person的装备
            pred_equipment = analyze_person_equipment(pred_person["bbox"], pred_helmets, pred_vests, overlap_threshold)
            unmatched_pred_persons.append({"person": pred_person, "equipment": pred_equipment})

    # 分析未匹配的真实person
    unmatched_gt_persons_with_equipment = []
    for gt_person in unmatched_gt_persons:
        gt_equipment = analyze_person_equipment(gt_person["bbox"], gt_helmets, gt_vests, overlap_threshold)
        unmatched_gt_persons_with_equipment.append({"person": gt_person, "equipment": gt_equipment})

    # 找出独立的helmet和vest（不属于任何person的）
    # 使用bbox坐标作为唯一标识，而不是id()
    def bbox_key(item):
        return tuple(item["bbox"])

    used_helmets = set()
    used_vests = set()

    # 收集所有已被person使用的helmet和vest的bbox
    for match in matched_persons:
        for helmet in match["pred_equipment"]["helmets"]:
            used_helmets.add(bbox_key(helmet))
        for vest in match["pred_equipment"]["vests"]:
            used_vests.add(bbox_key(vest))
        for helmet in match["gt_equipment"]["helmets"]:
            used_helmets.add(bbox_key(helmet))
        for vest in match["gt_equipment"]["vests"]:
            used_vests.add(bbox_key(vest))

    for unmatched in unmatched_pred_persons:
        for helmet in unmatched["equipment"]["helmets"]:
            used_helmets.add(bbox_key(helmet))
        for vest in unmatched["equipment"]["vests"]:
            used_vests.add(bbox_key(vest))

    for unmatched in unmatched_gt_persons_with_equipment:
        for helmet in unmatched["equipment"]["helmets"]:
            used_helmets.add(bbox_key(helmet))
        for vest in unmatched["equipment"]["vests"]:
            used_vests.add(bbox_key(vest))

    # 独立的helmet和vest（可能是真正的错误检测）
    independent_pred_helmets = [h for h in pred_helmets if bbox_key(h) not in used_helmets]
    independent_pred_vests = [v for v in pred_vests if bbox_key(v) not in used_vests]
    independent_gt_helmets = [h for h in gt_helmets if bbox_key(h) not in used_helmets]
    independent_gt_vests = [v for v in gt_vests if bbox_key(v) not in used_vests]

    return {
        "matched_persons": matched_persons,
        "unmatched_pred_persons": unmatched_pred_persons,
        "unmatched_gt_persons": unmatched_gt_persons_with_equipment,
        "independent_pred_helmets": independent_pred_helmets,
        "independent_pred_vests": independent_pred_vests,
        "independent_gt_helmets": independent_gt_helmets,
        "independent_gt_vests": independent_gt_vests,
    }


def calculate_equipment_statistics(analysis_result):
    """计算装备穿戴统计."""
    stats = {
        "person_detection": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
        "equipment_analysis": {
            "fully_equipped": {"correct": 0, "incorrect": 0},
            "helmet_only": {"correct": 0, "incorrect": 0},
            "vest_only": {"correct": 0, "incorrect": 0},
            "no_equipment": {"correct": 0, "incorrect": 0},
        },
        "component_detection": {"helmet": {"tp": 0, "fp": 0, "fn": 0}, "safety_vest": {"tp": 0, "fp": 0, "fn": 0}},
    }

    # Person检测统计
    stats["person_detection"]["true_positives"] = len(analysis_result["matched_persons"])
    stats["person_detection"]["false_positives"] = len(analysis_result["unmatched_pred_persons"])
    stats["person_detection"]["false_negatives"] = len(analysis_result["unmatched_gt_persons"])

    # 装备状态分析
    for match in analysis_result["matched_persons"]:
        pred_status = match["pred_equipment"]["equipment_status"]
        gt_status = match["gt_equipment"]["equipment_status"]

        if pred_status == gt_status:
            stats["equipment_analysis"][pred_status]["correct"] += 1
        else:
            stats["equipment_analysis"][pred_status]["incorrect"] += 1

    # 组件检测统计（独立的helmet和vest）
    stats["component_detection"]["helmet"]["fp"] = len(analysis_result["independent_pred_helmets"])
    stats["component_detection"]["helmet"]["fn"] = len(analysis_result["independent_gt_helmets"])
    stats["component_detection"]["safety_vest"]["fp"] = len(analysis_result["independent_pred_vests"])
    stats["component_detection"]["safety_vest"]["fn"] = len(analysis_result["independent_gt_vests"])

    return stats


def draw_hierarchical_analysis(image, analysis_result, class_names):
    """绘制层次化分析结果."""
    img_copy = image.copy()

    # 定义颜色
    colors = {
        "person_correct": (0, 255, 0),  # 绿色 - 正确的person
        "person_fp": (0, 0, 255),  # 红色 - 错误的person
        "person_fn": (255, 0, 0),  # 蓝色 - 漏检的person
        "helmet": (255, 255, 0),  # 青色 - helmet
        "safety_vest": (255, 0, 255),  # 紫色 - safety_vest
        "equipment_correct": (0, 255, 0),  # 绿色 - 装备状态正确
        "equipment_wrong": (0, 0, 255),  # 红色 - 装备状态错误
    }

    # 绘制匹配的person
    for match in analysis_result["matched_persons"]:
        pred_person = match["prediction"]
        pred_status = match["pred_equipment"]["equipment_status"]
        gt_status = match["gt_equipment"]["equipment_status"]

        # 选择颜色
        color = colors["equipment_correct"] if pred_status == gt_status else colors["equipment_wrong"]

        # 绘制预测框
        x1, y1, x2, y2 = pred_person["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # 添加标签
        label = f"Person: {pred_status}"
        if pred_status != gt_status:
            label += f" (GT: {gt_status})"

        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制装备
        for helmet in match["pred_equipment"]["helmets"]:
            hx1, hy1, hx2, hy2 = helmet["bbox"]
            cv2.rectangle(img_copy, (hx1, hy1), (hx2, hy2), colors["helmet"], 1)
            cv2.putText(img_copy, "H", (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors["helmet"], 1)

        for vest in match["pred_equipment"]["vests"]:
            vx1, vy1, vx2, vy2 = vest["bbox"]
            cv2.rectangle(img_copy, (vx1, vy1), (vx2, vy2), colors["safety_vest"], 1)
            cv2.putText(img_copy, "V", (vx1, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors["safety_vest"], 1)

    # 绘制未匹配的预测person (False Positives)
    for unmatched in analysis_result["unmatched_pred_persons"]:
        person = unmatched["person"]
        status = unmatched["equipment"]["equipment_status"]

        x1, y1, x2, y2 = person["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), colors["person_fp"], 2)
        cv2.putText(
            img_copy, f"FP Person: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["person_fp"], 2
        )

    # 绘制未匹配的真实person (False Negatives)
    for unmatched in analysis_result["unmatched_gt_persons"]:
        person = unmatched["person"]
        status = unmatched["equipment"]["equipment_status"]

        x1, y1, x2, y2 = person["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), colors["person_fn"], 2)
        cv2.putText(
            img_copy, f"FN Person: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["person_fn"], 2
        )

    # 绘制独立的helmet和vest（可能是错误检测）
    for helmet in analysis_result["independent_pred_helmets"]:
        x1, y1, x2, y2 = helmet["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), colors["helmet"], 2)
        cv2.putText(img_copy, "Independent Helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["helmet"], 2)

    for vest in analysis_result["independent_pred_vests"]:
        x1, y1, x2, y2 = vest["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), colors["safety_vest"], 2)
        cv2.putText(
            img_copy, "Independent Vest", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["safety_vest"], 2
        )

    return img_copy


def process_dataset_hierarchical(
    model, images_path, labels_path, output_dir, split_name, class_names, conf_threshold=0.25
):
    """处理数据集的层次化分析."""
    print(f"\n处理{split_name}...")

    # 创建输出目录
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    # 创建子目录
    error_dirs = {
        "person_detection_errors": os.path.join(split_output_dir, "person_detection_errors"),
        "equipment_status_errors": os.path.join(split_output_dir, "equipment_status_errors"),
        "independent_components": os.path.join(split_output_dir, "independent_components"),
        "correct_predictions": os.path.join(split_output_dir, "correct_predictions"),
    }

    for dir_path in error_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图像文件
    image_files = glob.glob(os.path.join(images_path, "*.jpg"))

    all_results = []
    overall_stats = {
        "person_detection": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
        "equipment_analysis": {
            "fully_equipped": {"correct": 0, "incorrect": 0},
            "helmet_only": {"correct": 0, "incorrect": 0},
            "vest_only": {"correct": 0, "incorrect": 0},
            "no_equipment": {"correct": 0, "incorrect": 0},
        },
        "component_detection": {"helmet": {"tp": 0, "fp": 0, "fn": 0}, "safety_vest": {"tp": 0, "fp": 0, "fn": 0}},
    }

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

        # 层次化分析
        analysis_result = hierarchical_match_analysis(predictions, ground_truth, class_names)

        # 计算统计信息
        img_stats = calculate_equipment_statistics(analysis_result)

        # 累加到总体统计
        for key in overall_stats["person_detection"]:
            overall_stats["person_detection"][key] += img_stats["person_detection"][key]

        for status in overall_stats["equipment_analysis"]:
            for result_type in overall_stats["equipment_analysis"][status]:
                overall_stats["equipment_analysis"][status][result_type] += img_stats["equipment_analysis"][status][
                    result_type
                ]

        for component in overall_stats["component_detection"]:
            for metric in overall_stats["component_detection"][component]:
                overall_stats["component_detection"][component][metric] += img_stats["component_detection"][component][
                    metric
                ]

        # 判断是否有错误 - 使用更合理的标准
        # 主要关注Person检测错误，装备状态错误相对宽松
        has_person_errors = (
            img_stats["person_detection"]["false_positives"] > 0 or img_stats["person_detection"]["false_negatives"] > 0
        )

        has_equipment_errors = any(
            img_stats["equipment_analysis"][status]["incorrect"] > 0 for status in img_stats["equipment_analysis"]
        )

        has_component_errors = (
            len(analysis_result["independent_pred_helmets"]) > 0
            or len(analysis_result["independent_pred_vests"]) > 0
            or len(analysis_result["independent_gt_helmets"]) > 0
            or len(analysis_result["independent_gt_vests"]) > 0
        )

        # 修改错误判断逻辑：主要基于Person检测，装备状态错误不算致命错误
        has_errors = has_person_errors  # 只有Person检测错误才算真正的错误

        # 保存结果
        result_data = {
            "image_file": img_file,
            "image_name": img_name,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "analysis_result": analysis_result,
            "statistics": img_stats,
            "has_errors": has_errors,
        }
        all_results.append(result_data)

        # 绘制并保存图像
        annotated_img = draw_hierarchical_analysis(image, analysis_result, class_names)

        if has_errors:
            if has_person_errors:
                save_path = os.path.join(error_dirs["person_detection_errors"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

            if has_equipment_errors:
                save_path = os.path.join(error_dirs["equipment_status_errors"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

            if has_component_errors:
                save_path = os.path.join(error_dirs["independent_components"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)
        else:
            correct_count += 1
            if correct_count <= 20:  # 只保存前20个正确的示例
                save_path = os.path.join(error_dirs["correct_predictions"], f"{img_name}.jpg")
                cv2.imwrite(save_path, annotated_img)

    # 计算性能指标
    person_precision = (
        overall_stats["person_detection"]["true_positives"]
        / (overall_stats["person_detection"]["true_positives"] + overall_stats["person_detection"]["false_positives"])
        if (overall_stats["person_detection"]["true_positives"] + overall_stats["person_detection"]["false_positives"])
        > 0
        else 0
    )

    person_recall = (
        overall_stats["person_detection"]["true_positives"]
        / (overall_stats["person_detection"]["true_positives"] + overall_stats["person_detection"]["false_negatives"])
        if (overall_stats["person_detection"]["true_positives"] + overall_stats["person_detection"]["false_negatives"])
        > 0
        else 0
    )

    person_f1 = (
        2 * (person_precision * person_recall) / (person_precision + person_recall)
        if (person_precision + person_recall) > 0
        else 0
    )

    # 计算装备状态准确率
    equipment_accuracy = {}
    for status in overall_stats["equipment_analysis"]:
        total = (
            overall_stats["equipment_analysis"][status]["correct"]
            + overall_stats["equipment_analysis"][status]["incorrect"]
        )
        if total > 0:
            equipment_accuracy[status] = overall_stats["equipment_analysis"][status]["correct"] / total
        else:
            equipment_accuracy[status] = 0

    # 计算no helmet和no vest的召回率
    # no helmet召回率：真实无helmet的人被正确识别为无helmet的比例
    # 允许在vest_only和no_equipment之间预测错误
    no_helmet_gt = (
        overall_stats["equipment_analysis"]["vest_only"]["correct"]
        + overall_stats["equipment_analysis"]["vest_only"]["incorrect"]
        + overall_stats["equipment_analysis"]["no_equipment"]["correct"]
        + overall_stats["equipment_analysis"]["no_equipment"]["incorrect"]
    )

    no_helmet_correct = (
        overall_stats["equipment_analysis"]["vest_only"]["correct"]
        + overall_stats["equipment_analysis"]["no_equipment"]["correct"]
    )

    no_helmet_recall = no_helmet_correct / no_helmet_gt if no_helmet_gt > 0 else 0

    # no vest召回率：真实无vest的人被正确识别为无vest的比例
    # 允许在helmet_only和no_equipment之间预测错误
    no_vest_gt = (
        overall_stats["equipment_analysis"]["helmet_only"]["correct"]
        + overall_stats["equipment_analysis"]["helmet_only"]["incorrect"]
        + overall_stats["equipment_analysis"]["no_equipment"]["correct"]
        + overall_stats["equipment_analysis"]["no_equipment"]["incorrect"]
    )

    no_vest_correct = (
        overall_stats["equipment_analysis"]["helmet_only"]["correct"]
        + overall_stats["equipment_analysis"]["no_equipment"]["correct"]
    )

    no_vest_recall = no_vest_correct / no_vest_gt if no_vest_gt > 0 else 0

    # 保存统计信息
    final_stats = {
        "total_images": len(image_files),
        "correct_predictions": correct_count,
        "accuracy": correct_count / len(image_files) if image_files else 0,
        "person_detection": {
            **overall_stats["person_detection"],
            "precision": person_precision,
            "recall": person_recall,
            "f1_score": person_f1,
        },
        "equipment_analysis": overall_stats["equipment_analysis"],
        "equipment_accuracy": equipment_accuracy,
        "safety_compliance": {
            "no_helmet_recall": no_helmet_recall,
            "no_vest_recall": no_vest_recall,
            "no_helmet_total": no_helmet_gt,
            "no_vest_total": no_vest_gt,
            "no_helmet_correct": no_helmet_correct,
            "no_vest_correct": no_vest_correct,
        },
        "component_detection": overall_stats["component_detection"],
    }

    stats_path = os.path.join(split_output_dir, "hierarchical_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    # 保存详细结果
    results_path = os.path.join(split_output_dir, "detailed_hierarchical_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        # 简化结果以减少文件大小
        simplified_results = []
        for result in all_results:
            simplified_results.append(
                {
                    "image_name": result["image_name"],
                    "has_errors": result["has_errors"],
                    "num_predictions": len(result["predictions"]),
                    "num_ground_truth": len(result["ground_truth"]),
                    "person_matches": len(result["analysis_result"]["matched_persons"]),
                    "person_fp": len(result["analysis_result"]["unmatched_pred_persons"]),
                    "person_fn": len(result["analysis_result"]["unmatched_gt_persons"]),
                    "equipment_stats": result["statistics"]["equipment_analysis"],
                }
            )
        json.dump(simplified_results, f, indent=2, ensure_ascii=False)

    print(f"{split_name}处理完成:")
    print(f"  总图像数: {len(image_files)}")
    print(f"  正确预测: {correct_count}")
    print(f"  准确率: {correct_count / len(image_files) * 100:.2f}%")
    print(f"  Person检测 - P: {person_precision:.3f}, R: {person_recall:.3f}, F1: {person_f1:.3f}")
    print(f"  装备状态准确率: {equipment_accuracy}")
    print("  安全合规检测:")
    print(f"    No Helmet召回率: {no_helmet_recall:.3f} ({no_helmet_correct}/{no_helmet_gt})")
    print(f"    No Vest召回率: {no_vest_recall:.3f} ({no_vest_correct}/{no_vest_gt})")

    return final_stats, all_results


def main():
    parser = argparse.ArgumentParser(description="层次化安全装备检测分析")
    parser.add_argument("--weights", default="runs/rail_train300epoch/yolov5s_/weights/best.pt", help="模型权重路径")
    parser.add_argument("--data", default="data/railroad-worker-detection", help="数据集路径")
    parser.add_argument("--output", default="hierarchical_analysis", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.1, help="置信度阈值")
    parser.add_argument("--overlap", type=float, default=0.3, help="装备重叠阈值")

    args = parser.parse_args()

    print("层次化安全装备检测分析")
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

    # 验证类别配置
    required_classes = ["safety_vest", "helmet", "person"]
    for req_class in required_classes:
        if req_class not in class_names:
            print(f"错误: 数据集中缺少必需的类别 '{req_class}'")
            return

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
            stats, _ = process_dataset_hierarchical(
                model, images_path, labels_path, args.output, split_name, class_names, args.conf
            )
            all_stats[split_name] = stats
        else:
            print(f"跳过{split_name}: 路径不存在")

    # 保存总体统计
    overall_stats_path = os.path.join(args.output, "overall_hierarchical_statistics.json")
    with open(overall_stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n层次化分析完成! 结果保存在: {args.output}")
    print("=" * 50)

    # 打印总结
    print("\n分析总结:")
    for split_name, stats in all_stats.items():
        print(f"\n{split_name.upper()}:")
        print("  Person检测性能:")
        print(f"    Precision: {stats['person_detection']['precision']:.3f}")
        print(f"    Recall: {stats['person_detection']['recall']:.3f}")
        print(f"    F1-Score: {stats['person_detection']['f1_score']:.3f}")
        print("  装备状态准确率:")
        for status, acc in stats["equipment_accuracy"].items():
            print(f"    {status}: {acc:.3f}")
        print("  安全合规检测:")
        print(f"    No Helmet召回率: {stats['safety_compliance']['no_helmet_recall']:.3f}")
        print(f"    No Vest召回率: {stats['safety_compliance']['no_vest_recall']:.3f}")


if __name__ == "__main__":
    main()
