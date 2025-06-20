# -*- coding: utf-8 -*-
"""
模块化的Excel数据可视化分析工具
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from PIL import Image

# 导入自定义模块
from config import APP_CONFIG
from file_handler import FileProcessor
from ai_chat import ChatProcessor, ChatHistoryManager
from chart_generator import ChartGenerator, ZoomController
from ui_components import UIComponents, SessionManager

# 尝试导入LangChain相关库
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("⚠️ LangChain未安装，请运行: pip install langchain")


def main():
    """
    主函数 - 应用程序入口点
    """
    # 设置页面配置
    UIComponents.setup_page_config()
    
    # 初始化会话状态
    SessionManager.initialize_session_state()
    ZoomController.initialize_zoom_state()
    
    # 创建页面标题
    UIComponents.create_header()
    
    # 创建侧边栏API配置
    api_config = UIComponents.create_api_config_section()
    UIComponents.create_sidebar_info()
    
    # 主要内容区域
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📊 数据配置与清洗")
        
        # 文件上传区域
        uploaded_file, file_params = UIComponents.create_file_upload_section()
        
        if uploaded_file is not None:
            try:
                # 使用文件处理器读取数据
                file_processor = FileProcessor()
                data = file_processor.process_file_with_options(
                    uploaded_file, 
                    uploaded_file.name.split('.')[-1].lower(),
                    file_params['header_row']
                )
                
                if data is not None and not data.empty:
                    # 数据验证
                    if file_processor.validate_data(data):
                        # 保存数据到会话状态
                        SessionManager.save_data(data)
                        
                        # 数据预览和清洗
                        cleaned_data = UIComponents.create_data_preview_section(data)
                        
                        # 更新会话状态中的数据
                        SessionManager.save_data(cleaned_data)
                        
                        # 列选择和图表配置
                        selected_columns, chart_type, animation_frames = UIComponents.create_column_selection_section(cleaned_data)
                        
                        if selected_columns:
                            # 图表布局选择
                            layout_type = UIComponents.create_chart_layout_section()
                            
                            # 轴分配（如果需要）
                            axis_assignment = UIComponents.create_axis_assignment_section(selected_columns, layout_type)
                            
                            # 保存图表配置
                            st.session_state.chart_config = {
                                'columns': selected_columns,
                                'chart_type': chart_type,
                                'layout_type': layout_type,
                                'axis_assignment': axis_assignment,
                                'animation_frames': animation_frames
                            }
                    else:
                        st.error("❌ 数据验证失败")
                else:
                    st.error("❌ 无法读取文件或文件为空")
                    
            except Exception as e:
                st.error(f"❌ 文件处理错误: {str(e)}")
    
    with col2:
        st.subheader("📈 数据可视化")
        
        # 获取当前数据
        current_data = SessionManager.get_current_data()
        
        if current_data is not None and 'chart_config' in st.session_state:
            config = st.session_state.chart_config
            
            if config.get('columns'):
                try:
                    # 创建图表生成器
                    chart_generator = ChartGenerator()
                    
                    # 根据布局类型生成图表
                    if config['layout_type'] == 'single':
                        fig = chart_generator.create_single_axis_chart(
                            current_data, 
                            config['columns'], 
                            config['chart_type'],
                            animation_frames=config.get('animation_frames')
                        )
                    elif config['layout_type'] == 'dual':
                        fig = chart_generator.create_dual_axis_chart(
                            current_data,
                            config['axis_assignment']['primary'],
                            config['axis_assignment']['secondary']
                        )
                    elif config['layout_type'] == 'triple':
                        fig = chart_generator.create_triple_axis_chart(
                            current_data,
                            config['axis_assignment']['primary'],
                            config['axis_assignment']['secondary'],
                            config['axis_assignment']['third']
                        )
                    elif config['layout_type'] == 'subplot':
                        fig = chart_generator.create_subplot_charts(
                            current_data, 
                            config['columns'], 
                            chart_type=config['chart_type']
                        )
                    elif config['layout_type'] == 'compact':
                        fig = chart_generator.create_compact_subplot(
                            current_data, 
                            config['columns']
                        )
                    
                    # 缩放控制
                    if config['layout_type'] in ['dual', 'triple']:
                        st.subheader("🔍 图表缩放控制")
                        
                        zoom_configs = {}
                        
                        # 主轴缩放控制
                        if config['axis_assignment']['primary']:
                            primary_data = current_data[config['axis_assignment']['primary']].select_dtypes(include=['number'])
                            if not primary_data.empty:
                                primary_range = (primary_data.min().min(), primary_data.max().max())
                                auto, manual_range = ZoomController.create_zoom_controls('primary', primary_range)
                                zoom_configs['primary'] = {'auto': auto, 'range': manual_range}
                        
                        # 副轴缩放控制
                        if config['axis_assignment']['secondary']:
                            secondary_data = current_data[config['axis_assignment']['secondary']].select_dtypes(include=['number'])
                            if not secondary_data.empty:
                                secondary_range = (secondary_data.min().min(), secondary_data.max().max())
                                auto, manual_range = ZoomController.create_zoom_controls('secondary', secondary_range)
                                zoom_configs['secondary'] = {'auto': auto, 'range': manual_range}
                        
                        # 第三轴缩放控制
                        if config['layout_type'] == 'triple' and config['axis_assignment']['third']:
                            third_data = current_data[config['axis_assignment']['third']].select_dtypes(include=['number'])
                            if not third_data.empty:
                                third_range = (third_data.min().min(), third_data.max().max())
                                auto, manual_range = ZoomController.create_zoom_controls('third', third_range)
                                zoom_configs['third'] = {'auto': auto, 'range': manual_range}
                        
                        # 应用缩放配置
                        ZoomController.apply_zoom_to_figure(fig, zoom_configs)
                    
                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 如果是动态图表，添加导出功能
                    if chart_type == 'animated':
                        st.markdown("---")
                        st.subheader("📥 导出动态图表")
                        
                        # 导出格式选择
                        export_format = st.radio(
                            "选择导出格式：",
                            ["HTML (交互式)", "GIF (动图)"],
                            horizontal=True
                        )
                        
                        # 如果选择GIF格式，先显示生成模式选择
                        if export_format == "GIF (动图)":
                            # 导入必要的库
                            import tempfile
                            import os
                            
                            # 添加生成模式选择
                            st.subheader("🎛️ GIF生成模式选择")
                            
                            generation_mode = st.radio(
                                "选择生成模式（根据您的时间需求）：",
                                options=["闪电模式", "超快速预览", "标准质量", "高质量"],
                                index=0,  # 默认选择闪电模式
                                help="闪电模式：10-30秒极速生成，超快速模式：1-3分钟，标准模式：5-15分钟，高质量模式：15-45分钟"
                            )
                            
                            try:
                                total_frames = len(fig.frames)
                                
                                # 根据生成模式设置参数
                                if generation_mode == "闪电模式":
                                    # 极致优化：超低分辨率，极大帧间隔，最小缩放
                                    width, height, scale = 240, 160, 0.3
                                    if total_frames <= 10:
                                        frame_step = 1
                                        max_frames = 10
                                    elif total_frames <= 30:
                                        frame_step = 3
                                        max_frames = 10
                                    elif total_frames <= 60:
                                        frame_step = 6
                                        max_frames = 10
                                    elif total_frames <= 100:
                                        frame_step = 10
                                        max_frames = 10
                                    else:
                                        frame_step = max(total_frames // 8, 1)
                                        max_frames = 8
                                    duration = 500  # 更长的帧间隔
                                    mode_desc = "闪电模式"
                                    time_estimate = "10-30秒"
                                    
                                elif generation_mode == "超快速预览":
                                    # 超激进优化：低分辨率，大帧间隔
                                    width, height, scale = 320, 240, 0.4
                                    if total_frames <= 20:
                                        frame_step = 2
                                    elif total_frames <= 50:
                                        frame_step = 4
                                    elif total_frames <= 100:
                                        frame_step = 6
                                    else:
                                        frame_step = 8
                                    duration = 300
                                    mode_desc = "超快速预览模式"
                                    time_estimate = "1-3分钟"
                                    
                                elif generation_mode == "标准质量":
                                    # 平衡的优化设置
                                    width, height, scale = 480, 360, 0.6
                                    if total_frames <= 30:
                                        frame_step = 1
                                    elif total_frames <= 60:
                                        frame_step = 2
                                    elif total_frames <= 100:
                                        frame_step = 3
                                    else:
                                        frame_step = 4
                                    duration = 250
                                    mode_desc = "标准质量模式"
                                    time_estimate = "3-8分钟"
                                    
                                else:  # 高质量模式
                                    # 高质量设置
                                    width, height, scale = 800, 600, 1.0
                                    if total_frames <= 30:
                                        frame_step = 1
                                    elif total_frames <= 60:
                                        frame_step = 1
                                    elif total_frames <= 100:
                                        frame_step = 2
                                    else:
                                        frame_step = 3
                                    duration = 200
                                    mode_desc = "高质量模式"
                                    time_estimate = "10-30分钟"
                                
                                # 计算实际要处理的帧数
                                if generation_mode == "闪电模式":
                                    # 闪电模式：严格限制最大帧数
                                    temp_frames = list(range(0, total_frames, frame_step))
                                    selected_frames = temp_frames[:max_frames]  # 限制最大帧数
                                else:
                                    selected_frames = list(range(0, total_frames, frame_step))
                                actual_frame_count = len(selected_frames)
                                
                                # 显示详细的优化信息和预估时间
                                info_col1, info_col2 = st.columns(2)
                                with info_col1:
                                    st.info(f"📊 **{mode_desc}**\n\n"
                                            f"• 原始帧数: {total_frames}\n"
                                            f"• 优化后帧数: {actual_frame_count}\n"
                                            f"• 分辨率: {width}×{height}\n"
                                            f"• 帧间隔: 每{frame_step}帧取1帧")
                                with info_col2:
                                    # 根据帧数和模式给出建议
                                    if generation_mode == "闪电模式":
                                        recommendation = "⚡ 极速推荐！"
                                    elif generation_mode == "超快速预览" and total_frames > 30:
                                        recommendation = "✅ 推荐此模式"
                                    elif generation_mode == "标准质量" and total_frames <= 50:
                                        recommendation = "✅ 推荐此模式"
                                    elif generation_mode == "高质量" and total_frames <= 30:
                                        recommendation = "✅ 推荐此模式"
                                    else:
                                        recommendation = "⚡ 可以尝试"
                                    
                                    st.warning(f"⏱️ **预估生成时间**\n\n"
                                               f"• {time_estimate}\n"
                                               f"• 帧数减少: {int((1 - actual_frame_count/total_frames) * 100)}%\n"
                                               f"• 建议: {recommendation}")
                            
                            except Exception as e:
                                st.error(f"❌ 无法获取动画帧信息: {str(e)}")
                                generation_mode = None
                        
                        # 导出按钮
                        if st.button("📥 开始导出", type="primary", use_container_width=True):
                            try:
                                if export_format == "HTML (交互式)":
                                    # 生成动图HTML
                                    html_str = fig.to_html(include_plotlyjs='cdn', 
                                                          config={'displayModeBar': False,
                                                                'staticPlot': False})
                                    
                                    # 创建下载按钮
                                    st.download_button(
                                        label="💾 下载动态图表 (HTML)",
                                        data=html_str,
                                        file_name=f"动态图表_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                                        mime="text/html",
                                        use_container_width=True
                                    )
                                    
                                    st.success("✅ HTML动图已准备好下载！可在浏览器中播放交互式动画。")
                                    st.info("💡 提示：HTML文件包含完整的交互式动画，支持缩放、悬停等功能。")
                                    
                                else:  # GIF格式
                                    # 检查是否已选择生成模式
                                    if 'generation_mode' not in locals() or generation_mode is None:
                                        st.error("❌ 请先选择GIF生成模式")
                                    else:
                                        # 开始GIF生成流程
                                        st.session_state.gif_generating = True
                                        st.session_state.generation_mode = generation_mode
                                        st.session_state.mode_desc = mode_desc
                                        st.session_state.actual_frame_count = actual_frame_count
                                        st.session_state.selected_frames = selected_frames
                                        st.session_state.width = width
                                        st.session_state.height = height
                                        st.session_state.scale = scale
                                        st.session_state.duration = duration
                                        st.session_state.fig = fig
                                        st.rerun()  # 重新运行以显示进度界面
                            except Exception as e:
                                st.error(f"❌ 导出失败: {str(e)}")
                        
                    # GIF生成进度显示（在按钮外部，基于session_state状态）
                    if st.session_state.get('gif_generating', False):
                        st.info(f"🎬 正在生成 {st.session_state.get('mode_desc', 'GIF')} 动图...")
                        
                        # 创建进度条和状态文本
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 添加取消按钮
                        cancel_col1, cancel_col2 = st.columns([3, 1])
                        with cancel_col2:
                            if st.button("❌ 取消生成", use_container_width=True):
                                st.session_state.gif_generating = False
                                st.rerun()
                        
                        # 从session_state获取参数
                        generation_mode = st.session_state.get('generation_mode')
                        actual_frame_count = st.session_state.get('actual_frame_count')
                        selected_frames = st.session_state.get('selected_frames')
                        width = st.session_state.get('width')
                        height = st.session_state.get('height')
                        scale = st.session_state.get('scale')
                        duration = st.session_state.get('duration')
                        fig = st.session_state.get('fig')
                        
                        # 多线程优化：利用CPU多核并行生成图像帧
                        images = [None] * actual_frame_count  # 预分配列表，保持帧顺序
                        completed_frames = 0
                        lock = threading.Lock()  # 用于线程安全的计数更新
                        
                        def generate_frame(idx, frame_idx):
                            """生成单个图像帧的函数"""
                            nonlocal completed_frames
                            
                            # 检查是否被取消
                            if not st.session_state.get('gif_generating', True):
                                return None
                            
                            try:
                                # 创建单帧图表
                                frame = fig.frames[frame_idx]
                                single_frame_fig = fig.__class__(data=frame.data, layout=fig.layout)
                                
                                # 根据模式选择最优的图像生成参数
                                generation_mode = st.session_state.get('generation_mode', '标准质量')
                                
                                if generation_mode == "闪电模式":
                                    # 闪电模式：使用JPEG格式和最低质量设置
                                    img_bytes = single_frame_fig.to_image(
                                        format="jpeg", 
                                        width=width,
                                        height=height,
                                        scale=scale
                                    )
                                    # 直接从字节流创建PIL图像，跳过不必要的转换
                                    img_buffer = io.BytesIO(img_bytes)
                                    pil_image = Image.open(img_buffer)
                                    # 闪电模式直接使用原始模式，减少转换开销
                                    if pil_image.mode not in ['RGB', 'P']:
                                        pil_image = pil_image.convert('RGB')
                                else:
                                    # 其他模式使用PNG格式
                                    img_bytes = single_frame_fig.to_image(
                                        format="png", 
                                        width=width,
                                        height=height,
                                        scale=scale
                                    )
                                    # 直接从字节流创建PIL图像
                                    img_buffer = io.BytesIO(img_bytes)
                                    pil_image = Image.open(img_buffer)
                                    # 转换为RGB模式以确保GIF兼容性
                                    if pil_image.mode != 'RGB':
                                        pil_image = pil_image.convert('RGB')
                                
                                # 线程安全地更新计数（UI更新在主线程中进行）
                                with lock:
                                    completed_frames += 1
                                
                                return (idx, pil_image)
                            
                            except Exception as e:
                                st.error(f"帧 {idx} 生成失败: {str(e)}")
                                return None
                        
                        try:
                            # 使用线程池并行生成图像帧
                            # 根据CPU核心数调整线程数，最大使用8个线程避免过度竞争
                            max_workers = min(8, len(selected_frames))
                            main_thread_completed = 0  # 主线程维护的完成计数
                            
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                # 提交所有任务
                                future_to_idx = {
                                    executor.submit(generate_frame, idx, frame_idx): idx 
                                    for idx, frame_idx in enumerate(selected_frames)
                                }
                                
                                # 收集结果并保持顺序，同时在主线程中更新进度
                                for future in as_completed(future_to_idx):
                                    # 检查是否被取消
                                    if not st.session_state.get('gif_generating', True):
                                        # 取消所有未完成的任务
                                        for f in future_to_idx:
                                            f.cancel()
                                        break
                                    
                                    result = future.result()
                                    if result is not None:
                                        idx, pil_image = result
                                        images[idx] = pil_image
                                        
                                        # 在主线程中更新进度显示
                                        main_thread_completed += 1
                                        progress = main_thread_completed / actual_frame_count
                                        progress_bar.progress(progress)
                                        status_text.text(f"🎬 正在生成第 {main_thread_completed}/{actual_frame_count} 帧... ({st.session_state.get('mode_desc', 'GIF')}) [多线程加速 {max_workers}线程]")
                                 
                            # 过滤掉None值（失败或取消的帧）
                            images = [img for img in images if img is not None]
                             
                            # 检查是否完成或被取消
                            if st.session_state.get('gif_generating', True) and images:
                                # 更新状态
                                status_text.text("🔄 正在合成GIF动图...")
                                 
                                # 根据模式创建优化的GIF
                                gif_buffer = io.BytesIO()
                                generation_mode = st.session_state.get('generation_mode', '标准质量')
                                
                                if generation_mode == "闪电模式":
                                    # 闪电模式：极致压缩设置
                                    # 先将图像转换为调色板模式以减少颜色数
                                    optimized_images = []
                                    for img in images:
                                        # 转换为调色板模式，最多16色
                                        img_p = img.convert('P', palette=Image.ADAPTIVE, colors=16)
                                        optimized_images.append(img_p)
                                    
                                    optimized_images[0].save(
                                        gif_buffer,
                                        format='GIF',
                                        save_all=True,
                                        append_images=optimized_images[1:],
                                        duration=duration,
                                        loop=0,
                                        optimize=True,
                                        disposal=2  # 清除前一帧，减少文件大小
                                    )
                                else:
                                    # 其他模式使用标准设置
                                    images[0].save(
                                        gif_buffer,
                                        format='GIF',
                                        save_all=True,
                                        append_images=images[1:],
                                        duration=duration,  # 动态调整的持续时间
                                        loop=0,  # 无限循环
                                        optimize=True  # 优化文件大小
                                    )
                                gif_buffer.seek(0)
                                 
                                # 清除进度显示
                                progress_bar.empty()
                                status_text.empty()
                                 
                                # 显示文件大小信息
                                gif_size_mb = len(gif_buffer.getvalue()) / (1024 * 1024)
                                st.success(f"✅ {st.session_state.get('mode_desc', 'GIF')} GIF生成完成！文件大小: {gif_size_mb:.2f} MB")
                                 
                                # 创建下载按钮
                                st.download_button(
                                    label=f"💾 下载 {st.session_state.get('mode_desc', 'GIF')} 动态图表 (GIF)",
                                    data=gif_buffer.getvalue(),
                                    file_name=f"动态图表_{generation_mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.gif",
                                    mime="image/gif",
                                    use_container_width=True
                                )
                                 
                                # 显示模式特定的说明
                                if generation_mode == "超快速预览":
                                    st.info(f"💡 {st.session_state.get('mode_desc', 'GIF')}：采用极低分辨率({width}×{height})和大帧间隔优化，生成速度提升90%+，适合快速预览效果。")
                                elif generation_mode == "标准质量":
                                    st.info(f"💡 {st.session_state.get('mode_desc', 'GIF')}：平衡质量和速度，采用中等分辨率({width}×{height})和智能帧数控制。")
                                else:
                                    st.info(f"💡 {st.session_state.get('mode_desc', 'GIF')}：最高质量输出，采用高分辨率({width}×{height})，适合最终发布使用。")
                                 
                                # 重置生成状态
                                st.session_state.gif_generating = False
                            elif st.session_state.get('gif_generating', True) and not images:
                                # 所有帧都生成失败或被取消了
                                progress_bar.empty()
                                status_text.empty()
                                st.error("❌ GIF生成失败，所有帧未能成功生成或被取消。")
                                st.session_state.gif_generating = False
                            else:
                                # 生成被取消
                                if not st.session_state.get('gif_generating', True): # 只有在主动取消时才显示
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.warning("⚠️ GIF生成已取消")
                                st.session_state.gif_generating = False
                        
                        except ImportError:
                            st.error("❌ 缺少必要的依赖库。请安装：pip install kaleido pillow")
                            st.session_state.gif_generating = False # 确保出错时也重置状态
                        except Exception as gif_error:
                            st.error(f"❌ GIF生成失败: {str(gif_error)}")
                            st.info("💡 建议：如果GIF生成失败，可以选择HTML格式导出。")
                            st.session_state.gif_generating = False # 确保出错时也重置状态
                    
                except Exception as e:
                    st.error(f"❌ 图表生成错误: {str(e)}")
            else:
                st.info("📋 请选择要可视化的数据列")
        else:
            st.info("📁 请先上传数据文件")
    
    # Chat功能区域
    st.markdown("---")
    
    if current_data is not None:
        user_input, send_button, clear_button = UIComponents.create_chat_section()
        
        # 处理聊天交互
        if clear_button:
            SessionManager.clear_chat_history()
            st.rerun()
        
        if send_button and user_input.strip():
            if not api_config['api_key']:
                st.error("❌ 请在侧边栏配置API密钥")
            else:
                try:
                    # 创建AI聊天实例
                    chat_processor = ChatProcessor()
                    
                    # 处理聊天请求
                    chat_response = chat_processor.process_chat_input(
                        user_input=user_input,
                        data=current_data,
                        deepseek_api_key=api_config['api_key']
                    )
                    response = chat_response['content']
                    
                    # 更新聊天历史
                    SessionManager.update_chat_history(user_input, response)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 处理请求时出错: {str(e)}")
    else:
        st.info("💬 上传数据后即可使用Chat功能")


if __name__ == "__main__":
    main()