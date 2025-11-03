#!/usr/bin/env python3
"""Verify the created Excel file and display its contents."""

import os

import pandas as pd


def verify_excel_file(excel_path):
    """Verify and display the contents of the Excel file."""
    if not os.path.exists(excel_path):
        print(f"Error: Excel file {excel_path} not found!")
        return

    try:
        # Read both sheets
        overall_df = pd.read_excel(excel_path, sheet_name="整体模型性能对比")
        no_vest_df = pd.read_excel(excel_path, sheet_name="NO-Safety Vest性能对比")

        print("=" * 80)
        print("Excel文件验证结果")
        print("=" * 80)
        print(f"文件路径: {excel_path}")
        print(f"文件大小: {os.path.getsize(excel_path)} bytes")

        print("\n工作表信息:")
        print("- 整体模型性能对比")
        print("- NO-Safety Vest性能对比")

        print("\n" + "=" * 80)
        print(f"整体模型性能对比 (共 {len(overall_df)} 个模型)")
        print("=" * 80)
        print(overall_df.to_string(index=False, float_format="%.4f"))

        print("\n" + "=" * 80)
        print(f"NO-Safety Vest性能对比 (共 {len(no_vest_df)} 个模型)")
        print("=" * 80)
        print(no_vest_df.to_string(index=False, float_format="%.4f"))

        print("\n" + "=" * 80)
        print("数据统计摘要")
        print("=" * 80)

        # Overall performance statistics
        print("\n整体模型性能统计:")
        print(
            f"最高 Precision: {overall_df['Precision'].max():.4f} ({overall_df.loc[overall_df['Precision'].idxmax(), '模型名称']})"
        )
        print(
            f"最高 Recall: {overall_df['Recall'].max():.4f} ({overall_df.loc[overall_df['Recall'].idxmax(), '模型名称']})"
        )
        print(
            f"最高 mAP@0.5: {overall_df['mAP@0.5'].max():.4f} ({overall_df.loc[overall_df['mAP@0.5'].idxmax(), '模型名称']})"
        )
        print(
            f"最高 mAP@0.5:0.95: {overall_df['mAP@0.5:0.95'].max():.4f} ({overall_df.loc[overall_df['mAP@0.5:0.95'].idxmax(), '模型名称']})"
        )
        print(
            f"最高 RA-mAP: {overall_df['RA-mAP'].max():.4f} ({overall_df.loc[overall_df['RA-mAP'].idxmax(), '模型名称']})"
        )

        # NO-Safety Vest performance statistics
        print("\nNO-Safety Vest性能统计:")
        print(
            f"最高 Precision: {no_vest_df['Precision'].max():.4f} ({no_vest_df.loc[no_vest_df['Precision'].idxmax(), '模型名称']})"
        )
        print(
            f"最高 Recall: {no_vest_df['Recall'].max():.4f} ({no_vest_df.loc[no_vest_df['Recall'].idxmax(), '模型名称']})"
        )
        print(
            f"最高 mAP@0.5: {no_vest_df['mAP@0.5'].max():.4f} ({no_vest_df.loc[no_vest_df['mAP@0.5'].idxmax(), '模型名称']})"
        )
        print(
            f"最高 mAP@0.5:0.95: {no_vest_df['mAP@0.5:0.95'].max():.4f} ({no_vest_df.loc[no_vest_df['mAP@0.5:0.95'].idxmax(), '模型名称']})"
        )

        print("\n" + "=" * 80)
        print("转换完成！Excel文件已成功创建并验证。")
        print("=" * 80)

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        import traceback

        traceback.print_exc()


def main():
    excel_path = "runs/sv6_train1000epoch__test_best_20250707_150649/summary_report.xlsx"
    verify_excel_file(excel_path)


if __name__ == "__main__":
    main()
