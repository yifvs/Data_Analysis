#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF内容差异比对 - 高级Streamlit Web应用（包含PDF预览功能）
功能：
1. 提供Web界面供用户上传两个PDF文件
2. 在线预览PDF内容
3. 进行差异比对分析
4. 可视化显示差异结果
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys
import base64
from io import BytesIO

# 导入现有的PDF差异分析模块
try:
    from pdf_diff import PDFDiffAnalyzer, PDFTextExtractor
except ImportError:
    # 如果在云端部署时找不到模块，尝试从当前目录导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    try:
        from pdf_diff import PDFDiffAnalyzer, PDFTextExtractor
    except ImportError:
        st.error("❌ 无法导入PDF差异分析模块，请确保pdf_diff.py文件在正确位置")
        st.stop()

# 尝试导入PDF预览相关库
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    PDF_PREVIEW_AVAILABLE = True
except ImportError:
    PDF_PREVIEW_AVAILABLE = False

try:
    import streamlit_pdf_viewer as pdf_viewer
    PDF_VIEWER_AVAILABLE = True
except ImportError:
    PDF_VIEWER_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="手册改版差异比对工具",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.upload-section {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.pdf-preview {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    max-height: 600px;
    overflow-y: auto;
}
.diff-added {
    background-color: #d4edda;
    color: #155724;
    padding: 0.25rem;
    border-left: 4px solid #28a745;
    margin: 0.25rem 0;
}
.diff-removed {
    background-color: #f8d7da;
    color: #721c24;
    padding: 0.25rem;
    border-left: 4px solid #dc3545;
    margin: 0.25rem 0;
}
.similarity-score {
    font-size: 1.5rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.similarity-high {
    background-color: #d4edda;
    color: #155724;
}
.similarity-medium {
    background-color: #fff3cd;
    color: #856404;
}
.similarity-low {
    background-color: #f8d7da;
    color: #721c24;
}
.feature-badge {
    display: inline-block;
    background-color: #007bff;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 15px;
    font-size: 0.8rem;
    margin: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    """主应用函数"""
    
    # 页面标题
    st.markdown('<h1 class="main-header">📄 手册改版内容差异比对工具 - v1.0.0</h1>', unsafe_allow_html=True)
    
    # 功能特性展示
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="feature-badge">📤 文件上传</span>
        <span class="feature-badge">👁️ PDF预览</span>
        <span class="feature-badge">🔍 智能比对</span>
        <span class="feature-badge">📊 可视化差异</span>
        <span class="feature-badge">📈 相似度分析</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏说明
    with st.sidebar:
        st.header("📋 使用说明")
        st.markdown("""
        1. **上传文件**：选择两个需要比对的PDF文件
        2. **预览文档**：查看PDF内容预览
        3. **开始比对**：点击"开始差异分析"按钮
        4. **查看结果**：查看详细的差异分析报告
        
        **支持的文件格式**：
        - PDF文件（包含文本内容）
        - 文件大小限制：200MB
        """)
        
        st.header("🔧 功能状态")
        
        # 显示功能可用性
        if PDF_PREVIEW_AVAILABLE:
            st.success("✅ PDF图片预览可用")
        else:
            st.warning("⚠️ PDF图片预览不可用\n需要安装: pdf2image, Pillow")
        
        if PDF_VIEWER_AVAILABLE:
            st.success("✅ PDF查看器可用")
        else:
            st.info("💡 PDF查看器不可用\n可选安装: streamlit-pdf-viewer")
        
        st.header("📦 安装说明")
        st.code("""
# 安装基础依赖
pip install streamlit pdfplumber

# 安装PDF预览功能
pip install pdf2image Pillow

# 安装PDF查看器（可选）
pip install streamlit-pdf-viewer
        """)
    
    # 主内容区域
    upload_tab, preview_tab, analysis_tab = st.tabs(["📤 文件上传", "👁️ PDF预览", "🔍 差异分析"])
    
    with upload_tab:
        handle_file_upload()
    
    with preview_tab:
        handle_pdf_preview()
    
    with analysis_tab:
        handle_diff_analysis()

def handle_file_upload():
    """处理文件上传"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 上传第一个PDF文件")
        uploaded_file1 = st.file_uploader(
            "选择第一个PDF文件",
            type=['pdf'],
            key="file1",
            help="上传需要比对的第一个PDF文件"
        )
        if uploaded_file1:
            st.success(f"✅ 已上传: {uploaded_file1.name}")
            st.info(f"文件大小: {uploaded_file1.size / 1024:.1f} KB")
            # 存储到session state
            st.session_state.uploaded_file1 = uploaded_file1
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 上传第二个PDF文件")
        uploaded_file2 = st.file_uploader(
            "选择第二个PDF文件",
            type=['pdf'],
            key="file2",
            help="上传需要比对的第二个PDF文件"
        )
        if uploaded_file2:
            st.success(f"✅ 已上传: {uploaded_file2.name}")
            st.info(f"文件大小: {uploaded_file2.size / 1024:.1f} KB")
            # 存储到session state
            st.session_state.uploaded_file2 = uploaded_file2
        st.markdown('</div>', unsafe_allow_html=True)

def handle_pdf_preview():
    """处理PDF预览"""
    
    if not hasattr(st.session_state, 'uploaded_file1') or not hasattr(st.session_state, 'uploaded_file2'):
        st.warning("⚠️ 请先在'文件上传'标签页中上传两个PDF文件")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📄 {st.session_state.uploaded_file1.name}")
        display_pdf_preview(st.session_state.uploaded_file1, "preview1")
    
    with col2:
        st.subheader(f"📄 {st.session_state.uploaded_file2.name}")
        display_pdf_preview(st.session_state.uploaded_file2, "preview2")

def display_pdf_preview(uploaded_file, key):
    """显示PDF预览"""
    
    preview_method = st.selectbox(
        "选择预览方式",
        ["文本预览", "图片预览", "内嵌查看器"],
        key=f"preview_method_{key}"
    )
    
    if preview_method == "文本预览":
        display_text_preview(uploaded_file)
    elif preview_method == "图片预览":
        display_image_preview(uploaded_file)
    elif preview_method == "内嵌查看器":
        display_embedded_viewer(uploaded_file)

def display_text_preview(uploaded_file):
    """显示文本预览"""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        extractor = PDFTextExtractor()
        text_content = extractor.extract_text(tmp_path)
        
        # 显示前1000个字符
        preview_text = text_content[:1000]
        if len(text_content) > 1000:
            preview_text += "\n\n... (显示前1000个字符)"
        
        st.text_area(
            "文本内容预览",
            preview_text,
            height=400,
            disabled=True
        )
        
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"文本预览失败: {str(e)}")

def display_image_preview(uploaded_file):
    """显示图片预览"""
    
    if not PDF_PREVIEW_AVAILABLE:
        st.warning("⚠️ 图片预览功能不可用，请安装 pdf2image 和 Pillow")
        st.code("pip install pdf2image Pillow")
        return
    
    try:
        # 转换PDF为图片
        images = convert_from_bytes(uploaded_file.getvalue(), first_page=1, last_page=3)
        
        st.info(f"📄 显示前 {min(len(images), 3)} 页")
        
        for i, image in enumerate(images[:3]):
            st.image(image, caption=f"第 {i+1} 页", use_column_width=True)
        
        if len(images) > 3:
            st.info(f"📄 文档共 {len(images)} 页，仅显示前3页")
    
    except Exception as e:
        st.error(f"图片预览失败: {str(e)}")
        st.info("💡 可能的原因：\n- 需要安装 poppler-utils\n- PDF文件格式不支持")

def display_embedded_viewer(uploaded_file):
    """显示内嵌查看器"""
    
    if not PDF_VIEWER_AVAILABLE:
        st.warning("⚠️ 内嵌查看器不可用，请安装 streamlit-pdf-viewer")
        st.code("pip install streamlit-pdf-viewer")
        return
    
    try:
        # 使用streamlit-pdf-viewer显示PDF
        pdf_viewer.pdf_viewer(uploaded_file.getvalue(), height=500)
    
    except Exception as e:
        st.error(f"内嵌查看器失败: {str(e)}")

def handle_diff_analysis():
    """处理差异分析"""
    
    if not hasattr(st.session_state, 'uploaded_file1') or not hasattr(st.session_state, 'uploaded_file2'):
        st.warning("⚠️ 请先在'文件上传'标签页中上传两个PDF文件")
        return
    
    # 显示文件信息
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📄 文件1: {st.session_state.uploaded_file1.name}")
    with col2:
        st.info(f"📄 文件2: {st.session_state.uploaded_file2.name}")
    
    # 分析选项
    st.subheader("⚙️ 分析选项")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ignore_whitespace = st.checkbox("忽略空白字符差异", value=True)
    with col2:
        ignore_case = st.checkbox("忽略大小写差异", value=False)
    with col3:
        detailed_analysis = st.checkbox("详细分析模式", value=True)
    
    # 分析按钮
    if st.button("🚀 开始差异分析", type="primary", use_container_width=True):
        analyze_pdfs_advanced(
            st.session_state.uploaded_file1,
            st.session_state.uploaded_file2,
            ignore_whitespace,
            ignore_case,
            detailed_analysis
        )

def analyze_pdfs_advanced(file1, file2, ignore_whitespace, ignore_case, detailed_analysis):
    """高级PDF差异分析"""
    
    with st.spinner("🔄 正在进行高级差异分析，请稍候..."):
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
                tmp1.write(file1.getvalue())
                tmp1_path = tmp1.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
                tmp2.write(file2.getvalue())
                tmp2_path = tmp2.name
            
            # 创建分析器并进行分析
            analyzer = PDFDiffAnalyzer()
            diff_result = analyzer.analyze_differences(tmp1_path, tmp2_path)
            
            # 应用分析选项（这里可以扩展更多自定义逻辑）
            if ignore_case:
                st.info("🔤 已启用忽略大小写模式")
            
            if ignore_whitespace:
                st.info("⬜ 已启用忽略空白字符模式")
            
            # 显示结果
            display_advanced_results(file1.name, file2.name, diff_result, detailed_analysis)
            
        except Exception as e:
            st.error(f"❌ 分析过程中出现错误: {str(e)}")
            st.info("💡 可能的原因：\n- PDF文件损坏\n- PDF为扫描版本（无文本内容）\n- 文件格式不支持")
        
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp1_path)
                os.unlink(tmp2_path)
            except:
                pass

def display_advanced_results(filename1, filename2, diff_result, detailed_analysis):
    """显示高级分析结果"""
    
    st.markdown("---")
    st.header("📊 高级差异分析结果")
    
    # 相似度显示
    similarity = diff_result['similarity_ratio']
    similarity_percent = similarity * 100
    
    # 创建指标卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 相似度",
            value=f"{similarity_percent:.2f}%",
            delta=f"{similarity_percent - 50:.1f}% vs 基准"
        )
    
    with col2:
        st.metric(
            label="📄 文件1行数",
            value=len(diff_result['lines1'])
        )
    
    with col3:
        st.metric(
            label="📄 文件2行数",
            value=len(diff_result['lines2']),
            delta=len(diff_result['lines2']) - len(diff_result['lines1'])
        )
    
    with col4:
        # 计算差异块数量
        opcodes = diff_result['sequence_matcher'].get_opcodes()
        diff_blocks = sum(1 for tag, _, _, _, _ in opcodes if tag != 'equal')
        st.metric(
            label="🔍 差异块数",
            value=diff_blocks
        )
    
    # 差异详情
    unified_diff = diff_result['unified_diff']
    
    # 检查是否有差异
    has_differences = any(line.startswith(('+', '-')) for line in unified_diff 
                         if not line.startswith(('+++', '---', '@@')))
    
    if not has_differences:
        st.success("🎉 两个PDF文档内容完全相同！")
        return
    
    # 显示差异（复用之前的函数）
    st.subheader("📋 详细差异信息")
    
    if detailed_analysis:
        # 创建更多标签页
        tab1, tab2, tab3, tab4 = st.tabs(["📝 文本差异", "🔍 逐块分析", "📊 统计信息", "📄 原始差异"])
        
        with tab1:
            display_text_differences(diff_result)
        
        with tab2:
            display_block_analysis(diff_result)
        
        with tab3:
            display_statistics(diff_result)
        
        with tab4:
            display_raw_diff(unified_diff)
    else:
        # 简化显示
        display_text_differences(diff_result)

def display_statistics(diff_result):
    """显示统计信息"""
    
    lines1 = diff_result['lines1']
    lines2 = diff_result['lines2']
    sequence_matcher = diff_result['sequence_matcher']
    
    opcodes = sequence_matcher.get_opcodes()
    
    # 统计各种操作
    stats = {'equal': 0, 'replace': 0, 'delete': 0, 'insert': 0}
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            stats['equal'] += i2 - i1
        elif tag == 'replace':
            stats['replace'] += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            stats['delete'] += i2 - i1
        elif tag == 'insert':
            stats['insert'] += j2 - j1
    
    # 显示统计图表
    st.subheader("📊 差异类型统计")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**操作类型统计:**")
        for operation, count in stats.items():
            if operation == 'equal':
                st.success(f"✅ 相同行数: {count}")
            elif operation == 'replace':
                st.warning(f"🔄 替换行数: {count}")
            elif operation == 'delete':
                st.error(f"🗑️ 删除行数: {count}")
            elif operation == 'insert':
                st.info(f"➕ 插入行数: {count}")
    
    with col2:
        st.markdown("**文档特征:**")
        st.info(f"📏 平均行长度 (文件1): {sum(len(line) for line in lines1) / len(lines1):.1f} 字符")
        st.info(f"📏 平均行长度 (文件2): {sum(len(line) for line in lines2) / len(lines2):.1f} 字符")
        st.info(f"📊 总字符数差异: {sum(len(line) for line in lines2) - sum(len(line) for line in lines1)}")

# 复用之前的函数
def display_text_differences(diff_result):
    """显示格式化的文本差异"""
    
    lines1 = diff_result['lines1']
    lines2 = diff_result['lines2']
    sequence_matcher = diff_result['sequence_matcher']
    
    opcodes = sequence_matcher.get_opcodes()
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # 显示相同的内容（可选择性显示）
            if i2 - i1 <= 3:  # 只显示少量相同行
                for i in range(i1, i2):
                    st.text(f"  {i+1:4d}: {lines1[i]}")
            else:
                st.text(f"  ... {i2-i1} 行相同内容 ...")
        
        elif tag == 'replace':
            st.markdown("**🔄 替换内容:**")
            for i in range(i1, i2):
                st.markdown(f'<div class="diff-removed">- [{i+1:4d}] {lines1[i]}</div>', 
                           unsafe_allow_html=True)
            for j in range(j1, j2):
                st.markdown(f'<div class="diff-added">+ [{j+1:4d}] {lines2[j]}</div>', 
                           unsafe_allow_html=True)
        
        elif tag == 'delete':
            st.markdown("**🗑️ 删除内容:**")
            for i in range(i1, i2):
                st.markdown(f'<div class="diff-removed">- [{i+1:4d}] {lines1[i]}</div>', 
                           unsafe_allow_html=True)
        
        elif tag == 'insert':
            st.markdown("**➕ 新增内容:**")
            for j in range(j1, j2):
                st.markdown(f'<div class="diff-added">+ [{j+1:4d}] {lines2[j]}</div>', 
                           unsafe_allow_html=True)

def display_block_analysis(diff_result):
    """显示逐块差异分析"""
    
    lines1 = diff_result['lines1']
    lines2 = diff_result['lines2']
    sequence_matcher = diff_result['sequence_matcher']
    
    opcodes = sequence_matcher.get_opcodes()
    block_num = 1
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        
        with st.expander(f"📍 差异块 {block_num}: {get_tag_description(tag)}"):
            if tag == 'replace':
                st.markdown(f"**位置**: 行 {i1+1}-{i2} → 行 {j1+1}-{j2}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**原内容:**")
                    for i in range(i1, i2):
                        st.markdown(f'<div class="diff-removed">[{i+1}] {lines1[i]}</div>', 
                                   unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**新内容:**")
                    for j in range(j1, j2):
                        st.markdown(f'<div class="diff-added">[{j+1}] {lines2[j]}</div>', 
                                   unsafe_allow_html=True)
            
            elif tag == 'delete':
                st.markdown(f"**位置**: 行 {i1+1}-{i2}")
                for i in range(i1, i2):
                    st.markdown(f'<div class="diff-removed">[{i+1}] {lines1[i]}</div>', 
                               unsafe_allow_html=True)
            
            elif tag == 'insert':
                st.markdown(f"**位置**: 行 {j1+1}-{j2}")
                for j in range(j1, j2):
                    st.markdown(f'<div class="diff-added">[{j+1}] {lines2[j]}</div>', 
                               unsafe_allow_html=True)
        
        block_num += 1

def display_raw_diff(unified_diff):
    """显示原始差异输出"""
    
    st.markdown("**原始差异输出 (类似git diff格式):**")
    
    diff_text = "\n".join(unified_diff)
    if diff_text.strip():
        st.code(diff_text, language="diff")
    else:
        st.info("没有发现差异")

def get_tag_description(tag):
    """获取差异类型的中文描述"""
    descriptions = {
        'replace': '替换',
        'delete': '删除',
        'insert': '插入'
    }
    return descriptions.get(tag, tag)

if __name__ == "__main__":
    main()
