#!/usr/bin/env python3
"""LaTeX文件语法验证脚本 检查LaTeX文件的基本语法错误."""

import re
import sys
from pathlib import Path


def validate_latex_syntax(file_path):
    """验证LaTeX文件的基本语法."""
    print(f"验证LaTeX文件: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    errors = []
    warnings = []

    # 检查基本结构
    if not re.search(r"\\documentclass", content):
        errors.append("缺少 \\documentclass 声明")

    if not re.search(r"\\begin{document}", content):
        errors.append("缺少 \\begin{document}")

    if not re.search(r"\\end{document}", content):
        errors.append("缺少 \\end{document}")

    # 检查环境匹配
    begin_matches = re.findall(r"\\begin{([^}]+)}", content)
    end_matches = re.findall(r"\\end{([^}]+)}", content)

    for env in begin_matches:
        if env not in end_matches:
            errors.append(f"环境 {env} 没有对应的 \\end{{{env}}}")

    for env in end_matches:
        if env not in begin_matches:
            errors.append(f"环境 {env} 没有对应的 \\begin{{{env}}}")

    # 检查括号匹配
    open_braces = content.count("{")
    close_braces = content.count("}")
    if open_braces != close_braces:
        errors.append(f"大括号不匹配: {open_braces} 个 '{{' vs {close_braces} 个 '}}'")

    # 检查数学环境
    math_envs = ["equation", "align", "gather", "multiline"]
    for env in math_envs:
        begin_count = len(re.findall(rf"\\begin{{{env}}}", content))
        end_count = len(re.findall(rf"\\end{{{env}}}", content))
        if begin_count != end_count:
            errors.append(f"数学环境 {env} 不匹配: {begin_count} begin vs {end_count} end")

    # 检查表格环境
    table_begin = len(re.findall(r"\\begin{table}", content))
    table_end = len(re.findall(r"\\end{table}", content))
    if table_begin != table_end:
        errors.append(f"table环境不匹配: {table_begin} begin vs {table_end} end")

    tabular_begin = len(re.findall(r"\\begin{tabular}", content))
    tabular_end = len(re.findall(r"\\end{tabular}", content))
    if tabular_begin != tabular_end:
        errors.append(f"tabular环境不匹配: {tabular_begin} begin vs {tabular_end} end")

    # 检查代码环境
    lstlisting_begin = len(re.findall(r"\\begin{lstlisting}", content))
    lstlisting_end = len(re.findall(r"\\end{lstlisting}", content))
    if lstlisting_begin != lstlisting_end:
        errors.append(f"lstlisting环境不匹配: {lstlisting_begin} begin vs {lstlisting_end} end")

    # 检查参考文献
    if "\\begin{thebibliography}" in content and "\\end{thebibliography}" not in content:
        errors.append("thebibliography环境没有正确结束")

    # 检查常见错误
    if re.search(r"\\cite{[^}]*}[^.,;:\s]", content):
        warnings.append("引用后可能缺少标点符号")

    if re.search(r"[^\\]%[^%]", content):
        warnings.append("发现可能的注释符号，请确认是否正确")

    # 输出结果
    print("\n" + "=" * 60)
    print("LaTeX语法验证结果")
    print("=" * 60)

    if not errors and not warnings:
        print("✅ 语法检查通过！没有发现错误或警告。")
    else:
        if errors:
            print(f"❌ 发现 {len(errors)} 个错误:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")

        if warnings:
            print(f"⚠️  发现 {len(warnings)} 个警告:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

    # 统计信息
    print("\n" + "-" * 40)
    print("文件统计信息:")
    print(f"  总行数: {len(content.splitlines())}")
    print(f"  总字符数: {len(content)}")

    # 避免在f-string中使用反斜杠
    section_pattern = r"\\section{"
    subsection_pattern = r"\\subsection{"
    subsubsection_pattern = r"\\subsubsection{"
    equation_pattern = r"\\begin{equation}"
    table_pattern = r"\\begin{table}"
    lstlisting_pattern = r"\\begin{lstlisting}"
    bibitem_pattern = r"\\bibitem{"

    print(f"  章节数: {len(re.findall(section_pattern, content))}")
    print(f"  子章节数: {len(re.findall(subsection_pattern, content))}")
    print(f"  子子章节数: {len(re.findall(subsubsection_pattern, content))}")
    print(f"  公式数: {len(re.findall(equation_pattern, content))}")
    print(f"  表格数: {len(re.findall(table_pattern, content))}")
    print(f"  代码块数: {len(re.findall(lstlisting_pattern, content))}")
    print(f"  参考文献数: {len(re.findall(bibitem_pattern, content))}")

    return len(errors) == 0


def main():
    """主函数."""
    if len(sys.argv) > 1:
        tex_file = sys.argv[1]
    else:
        tex_file = "yolov5_innovations.tex"

    tex_path = Path(tex_file)
    if not tex_path.exists():
        print(f"错误: 文件 {tex_file} 不存在")
        return False

    return validate_latex_syntax(tex_path)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
