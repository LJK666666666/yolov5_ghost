#!/usr/bin/env python3
"""
Convert YOLOv5 summary report tables to Excel format
"""

import pandas as pd
import re
import os

def parse_summary_report(file_path):
    """Parse the summary report and extract the two tables"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract overall performance table
    overall_start = content.find("整体模型性能对比:")
    overall_end = content.find("NO-Safety Vest 类别性能对比:")
    overall_section = content[overall_start:overall_end]
    
    # Extract NO-Safety Vest performance table
    no_vest_start = content.find("NO-Safety Vest 类别性能对比:")
    no_vest_end = content.find("最佳模型分析:")
    no_vest_section = content[no_vest_start:no_vest_end]
    
    # Parse overall performance table
    overall_data = []
    lines = overall_section.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('-') and not line.startswith('整体模型性能对比') and not line.startswith('模型名称'):
            # Split by multiple spaces to separate columns
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 6:  # Model name + 5 metrics
                model_name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                map_05 = float(parts[3])
                map_05_095 = float(parts[4])
                ra_map = float(parts[5])
                
                overall_data.append({
                    '模型名称': model_name,
                    'Precision': precision,
                    'Recall': recall,
                    'mAP@0.5': map_05,
                    'mAP@0.5:0.95': map_05_095,
                    'RA-mAP': ra_map
                })
    
    # Parse NO-Safety Vest performance table
    no_vest_data = []
    lines = no_vest_section.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('-') and not line.startswith('NO-Safety Vest') and not line.startswith('模型名称'):
            # Split by multiple spaces to separate columns
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 5:  # Model name + 4 metrics
                model_name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                map_05 = float(parts[3])
                map_05_095 = float(parts[4])
                
                no_vest_data.append({
                    '模型名称': model_name,
                    'Precision': precision,
                    'Recall': recall,
                    'mAP@0.5': map_05,
                    'mAP@0.5:0.95': map_05_095
                })
    
    return overall_data, no_vest_data

def create_excel_file(overall_data, no_vest_data, output_path):
    """Create Excel file with two sheets"""
    
    # Create DataFrames
    overall_df = pd.DataFrame(overall_data)
    no_vest_df = pd.DataFrame(no_vest_data)
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write overall performance sheet
        overall_df.to_excel(writer, sheet_name='整体模型性能对比', index=False)
        
        # Write NO-Safety Vest performance sheet
        no_vest_df.to_excel(writer, sheet_name='NO-Safety Vest性能对比', index=False)
        
        # Get workbook and worksheets for formatting
        workbook = writer.book
        overall_ws = writer.sheets['整体模型性能对比']
        no_vest_ws = writer.sheets['NO-Safety Vest性能对比']
        
        # Auto-adjust column widths
        for ws in [overall_ws, no_vest_ws]:
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 20)
                ws.column_dimensions[column_letter].width = adjusted_width

def main():
    # Input and output paths
    input_file = "runs/sv6_train1000epoch__test_best_20250707_150649/summary_report.txt"
    output_file = "runs/sv6_train1000epoch__test_best_20250707_150649/summary_report.xlsx"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    try:
        # Parse the summary report
        print("Parsing summary report...")
        overall_data, no_vest_data = parse_summary_report(input_file)
        
        print(f"Found {len(overall_data)} models in overall performance table")
        print(f"Found {len(no_vest_data)} models in NO-Safety Vest performance table")
        
        # Create Excel file
        print("Creating Excel file...")
        create_excel_file(overall_data, no_vest_data, output_file)
        
        print(f"Excel file created successfully: {output_file}")
        
        # Display summary
        print("\n整体模型性能对比 (前5行):")
        overall_df = pd.DataFrame(overall_data)
        print(overall_df.head().to_string(index=False))
        
        print("\nNO-Safety Vest性能对比 (前5行):")
        no_vest_df = pd.DataFrame(no_vest_data)
        print(no_vest_df.head().to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
