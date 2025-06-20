#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF内容差异比对工具
功能：
1. 提取两个PDF文档的文本内容
2. 进行逐行差异比对
3. 输出详细的差异信息和位置
"""

import sys
import os
import argparse
import difflib
from typing import List, Tuple, Dict
from pathlib import Path

try:
    import pdfplumber
    PDF_LIBRARY = 'pdfplumber'
except ImportError:
    try:
        import PyPDF2
        PDF_LIBRARY = 'PyPDF2'
    except ImportError:
        print("错误：请安装 pdfplumber 或 PyPDF2 库")
        print("运行：pip install pdfplumber 或 pip install PyPDF2")
        sys.exit(1)

try:
    from colorama import init, Fore, Back, Style
    init()  # 初始化colorama
    COLOR_SUPPORT = True
except ImportError:
    COLOR_SUPPORT = False
    print("警告：未安装colorama库，将使用无颜色输出")


class PDFTextExtractor:
    """PDF文本提取器"""
    
    def __init__(self):
        self.library = PDF_LIBRARY
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """使用pdfplumber提取PDF文本"""
        text_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- 第{page_num}页 ---")
                        text_content.append(text)
                    else:
                        text_content.append(f"--- 第{page_num}页 (无文本内容) ---")
        except Exception as e:
            raise Exception(f"使用pdfplumber提取PDF文本失败: {str(e)}")
        
        return '\n'.join(text_content)
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """使用PyPDF2提取PDF文本"""
        text_content = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- 第{page_num}页 ---")
                        text_content.append(text)
                    else:
                        text_content.append(f"--- 第{page_num}页 (无文本内容) ---")
        except Exception as e:
            raise Exception(f"使用PyPDF2提取PDF文本失败: {str(e)}")
        
        return '\n'.join(text_content)
    
    def extract_text(self, pdf_path: str) -> str:
        """提取PDF文本内容"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        print(f"正在提取PDF文本: {os.path.basename(pdf_path)}")
        
        if self.library == 'pdfplumber':
            return self.extract_text_pdfplumber(pdf_path)
        else:
            return self.extract_text_pypdf2(pdf_path)


class PDFDiffAnalyzer:
    """PDF差异分析器"""
    
    def __init__(self):
        self.extractor = PDFTextExtractor()
    
    def normalize_text(self, text: str) -> List[str]:
        """标准化文本，按行分割并清理"""
        lines = text.split('\n')
        # 移除空行和只包含空白字符的行
        normalized_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # 只保留非空行
                normalized_lines.append(stripped_line)
        return normalized_lines
    
    def analyze_differences(self, pdf1_path: str, pdf2_path: str) -> Dict:
        """分析两个PDF文件的差异"""
        # 提取文本内容
        try:
            text1 = self.extractor.extract_text(pdf1_path)
            text2 = self.extractor.extract_text(pdf2_path)
        except Exception as e:
            raise Exception(f"文本提取失败: {str(e)}")
        
        # 标准化文本
        lines1 = self.normalize_text(text1)
        lines2 = self.normalize_text(text2)
        
        print(f"\n文件1 ({os.path.basename(pdf1_path)}) 共 {len(lines1)} 行")
        print(f"文件2 ({os.path.basename(pdf2_path)}) 共 {len(lines2)} 行")
        
        # 使用difflib进行差异分析
        differ = difflib.unified_diff(
            lines1, 
            lines2, 
            fromfile=os.path.basename(pdf1_path),
            tofile=os.path.basename(pdf2_path),
            lineterm=''
        )
        
        # 详细差异分析
        sequence_matcher = difflib.SequenceMatcher(None, lines1, lines2)
        
        return {
            'unified_diff': list(differ),
            'sequence_matcher': sequence_matcher,
            'lines1': lines1,
            'lines2': lines2,
            'similarity_ratio': sequence_matcher.ratio()
        }
    
    def format_differences(self, diff_result: Dict) -> str:
        """格式化差异结果"""
        output = []
        
        # 相似度信息
        similarity = diff_result['similarity_ratio']
        output.append(f"\n{'='*60}")
        output.append(f"文档相似度: {similarity:.2%}")
        output.append(f"{'='*60}")
        
        # 统一差异格式输出
        unified_diff = diff_result['unified_diff']
        if not any(line.startswith(('+', '-')) for line in unified_diff if not line.startswith(('+++', '---', '@@'))):
            output.append("\n✅ 两个PDF文档内容完全相同！")
            return '\n'.join(output)
        
        output.append("\n📋 详细差异信息:")
        output.append("-" * 60)
        
        for line in unified_diff:
            if line.startswith('+++') or line.startswith('---'):
                continue
            elif line.startswith('@@'):
                if COLOR_SUPPORT:
                    output.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
                else:
                    output.append(line)
            elif line.startswith('+'):
                if COLOR_SUPPORT:
                    output.append(f"{Fore.GREEN}+ {line[1:]}{Style.RESET_ALL}")
                else:
                    output.append(f"+ {line[1:]}")
            elif line.startswith('-'):
                if COLOR_SUPPORT:
                    output.append(f"{Fore.RED}- {line[1:]}{Style.RESET_ALL}")
                else:
                    output.append(f"- {line[1:]}")
            else:
                output.append(f"  {line}")
        
        # 详细的逐块差异分析
        output.append("\n" + "="*60)
        output.append("🔍 逐块差异分析:")
        output.append("-" * 60)
        
        sequence_matcher = diff_result['sequence_matcher']
        lines1 = diff_result['lines1']
        lines2 = diff_result['lines2']
        
        opcodes = sequence_matcher.get_opcodes()
        block_num = 1
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                continue
            
            output.append(f"\n📍 差异块 {block_num}:")
            
            if tag == 'replace':
                output.append(f"   类型: 替换 (行 {i1+1}-{i2} → 行 {j1+1}-{j2})")
                output.append("   原内容:")
                for i in range(i1, i2):
                    if COLOR_SUPPORT:
                        output.append(f"     {Fore.RED}- [{i+1}] {lines1[i]}{Style.RESET_ALL}")
                    else:
                        output.append(f"     - [{i+1}] {lines1[i]}")
                output.append("   新内容:")
                for j in range(j1, j2):
                    if COLOR_SUPPORT:
                        output.append(f"     {Fore.GREEN}+ [{j+1}] {lines2[j]}{Style.RESET_ALL}")
                    else:
                        output.append(f"     + [{j+1}] {lines2[j]}")
            
            elif tag == 'delete':
                output.append(f"   类型: 删除 (行 {i1+1}-{i2})")
                for i in range(i1, i2):
                    if COLOR_SUPPORT:
                        output.append(f"     {Fore.RED}- [{i+1}] {lines1[i]}{Style.RESET_ALL}")
                    else:
                        output.append(f"     - [{i+1}] {lines1[i]}")
            
            elif tag == 'insert':
                output.append(f"   类型: 插入 (行 {j1+1}-{j2})")
                for j in range(j1, j2):
                    if COLOR_SUPPORT:
                        output.append(f"     {Fore.GREEN}+ [{j+1}] {lines2[j]}{Style.RESET_ALL}")
                    else:
                        output.append(f"     + [{j+1}] {lines2[j]}")
            
            block_num += 1
        
        return '\n'.join(output)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='PDF内容差异比对工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python pdf_diff.py file1.pdf file2.pdf
  python pdf_diff.py "文档1.pdf" "文档2.pdf" --output diff_result.txt
        """
    )
    
    parser.add_argument('pdf1', help='第一个PDF文件路径')
    parser.add_argument('pdf2', help='第二个PDF文件路径')
    parser.add_argument('--output', '-o', help='输出结果到文件')
    parser.add_argument('--no-color', action='store_true', help='禁用颜色输出')
    
    args = parser.parse_args()
    
    # 禁用颜色输出
    if args.no_color:
        global COLOR_SUPPORT
        COLOR_SUPPORT = False
    
    try:
        # 创建分析器
        analyzer = PDFDiffAnalyzer()
        
        print("🚀 开始PDF差异分析...")
        print(f"文件1: {args.pdf1}")
        print(f"文件2: {args.pdf2}")
        
        # 分析差异
        diff_result = analyzer.analyze_differences(args.pdf1, args.pdf2)
        
        # 格式化输出
        formatted_output = analyzer.format_differences(diff_result)
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                # 移除颜色代码用于文件输出
                clean_output = formatted_output
                if COLOR_SUPPORT:
                    import re
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    clean_output = ansi_escape.sub('', formatted_output)
                f.write(clean_output)
            print(f"\n💾 结果已保存到: {args.output}")
        
        print(formatted_output)
        
        print("\n✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()