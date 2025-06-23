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
    
    # 文件上传区域
    st.subheader("📂 文件上传与数据处理")
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
                else:
                    st.error("❌ 数据验证失败")
                    data = None  # 验证失败则不处理后续
            else:
                st.error("❌ 无法读取文件或文件为空")
                data = None

        except Exception as e:
            st.error(f"❌ 文件处理错误: {str(e)}")
            data = None
    else:
        data = None

    # 获取当前数据
    current_data = SessionManager.get_current_data()

    if current_data is not None:
        with st.expander("⚙️ 配置选项", expanded=True):
            # 数据预览和清洗
            cleaned_data = UIComponents.create_data_preview_section(current_data)
            
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
        
    # --- 数据可视化部分 ---
    current_data = SessionManager.get_current_data() # 重新获取可能已清洗的数据
    
    # 创建标签页 - 移到条件外部，确保tab2始终被定义
    st.markdown("--- ")
    st.subheader("📈 数据可视化")
    tab1, tab2 = st.tabs(["🎯 标准可视化", "🛠️ 自定义轴配置"])  
    
    if current_data is not None and 'chart_config' in st.session_state:
        with tab1:
            # 原有的标准可视化功能
            if 'chart_config' in st.session_state:
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
                        if 'chart_type' in config and config['chart_type'] == 'animated':
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
                                        # 极致优化：低分辨率，极大帧间隔，最小缩放
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
                                        
                                        st.warning(f"⏱️ **预估生成时间: {time_estimate}**\n\n"
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
                            
                            # GIF生成进度显示（在GIF格式选择内部，基于session_state状态）
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
                                images = [None] * actual_frame_count  # Pre-allocate list to maintain frame order
                                completed_frames = 0
                                lock = threading.Lock()  # Lock for thread-safe counter updates

                                def generate_frame(idx, frame_idx):
                                    """Function to generate a single image frame."""
                                    nonlocal completed_frames

                                    if not st.session_state.get('gif_generating', True):
                                        return None

                                    try:
                                        frame = fig.frames[frame_idx]
                                        single_frame_fig = fig.__class__(data=frame.data, layout=fig.layout)

                                        generation_mode = st.session_state.get('generation_mode', '标准质量')

                                        if generation_mode == "闪电模式":
                                            img_bytes = single_frame_fig.to_image(
                                                format="jpeg", width=width, height=height, scale=scale
                                            )
                                            pil_image = Image.open(io.BytesIO(img_bytes))
                                            if pil_image.mode not in ['RGB', 'P']:
                                                pil_image = pil_image.convert('RGB')
                                        else:
                                            img_bytes = single_frame_fig.to_image(
                                                format="png", width=width, height=height, scale=scale
                                            )
                                            pil_image = Image.open(io.BytesIO(img_bytes))
                                            if pil_image.mode != 'RGB':
                                                pil_image = pil_image.convert('RGB')

                                        with lock:
                                            completed_frames += 1

                                        return (idx, pil_image)

                                    except Exception as e:
                                        # Log error without stopping other threads
                                        return None

                                try:
                                    max_workers = min(8, len(selected_frames))
                                    main_thread_completed = 0

                                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                        future_to_idx = {
                                            executor.submit(generate_frame, idx, frame_idx): idx
                                            for idx, frame_idx in enumerate(selected_frames)
                                        }

                                        for future in as_completed(future_to_idx):
                                            if not st.session_state.get('gif_generating', True):
                                                for f in future_to_idx:
                                                    f.cancel()
                                                break

                                            result = future.result()
                                            if result is not None:
                                                idx, pil_image = result
                                                images[idx] = pil_image

                                                main_thread_completed += 1
                                                progress = main_thread_completed / actual_frame_count
                                                progress_bar.progress(progress)
                                                status_text.text(f"🎬 正在生成第 {main_thread_completed}/{actual_frame_count} 帧... ({st.session_state.get('mode_desc', 'GIF')}) [多线程加速 {max_workers}线程]")

                                    images = [img for img in images if img is not None]

                                    if st.session_state.get('gif_generating', True) and images:
                                        status_text.text("🔄 正在合成GIF动图...")
                                        gif_buffer = io.BytesIO()
                                        generation_mode = st.session_state.get('generation_mode', '标准质量')

                                        if generation_mode == "闪电模式":
                                            optimized_images = []
                                            for img in images:
                                                img_p = img.convert('P', palette=Image.ADAPTIVE, colors=16)
                                                optimized_images.append(img_p)

                                            optimized_images[0].save(
                                                gif_buffer, format='GIF', save_all=True,
                                                append_images=optimized_images[1:],
                                                duration=duration, loop=0, optimize=True, disposal=2
                                            )
                                        else:
                                            images[0].save(
                                                gif_buffer, format='GIF', save_all=True,
                                                append_images=images[1:],
                                                duration=duration, loop=0, optimize=True
                                            )
                                        gif_buffer.seek(0)

                                        progress_bar.empty()
                                        status_text.empty()

                                        gif_size_mb = len(gif_buffer.getvalue()) / (1024 * 1024)
                                        st.success(f"✅ {st.session_state.get('mode_desc', 'GIF')} GIF生成完成！文件大小: {gif_size_mb:.2f} MB")

                                        st.download_button(
                                            label=f"💾 下载 {st.session_state.get('mode_desc', 'GIF')} 动态图表 (GIF)",
                                            data=gif_buffer.getvalue(),
                                            file_name=f"动态图表_{generation_mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.gif",
                                            mime="image/gif",
                                            use_container_width=True
                                        )

                                        if generation_mode == "超快速预览":
                                            st.info(f"💡 {st.session_state.get('mode_desc', 'GIF')}：采用极低分辨率({width}×{height})和大帧间隔优化，生成速度提升90%+，适合快速预览效果。")
                                        elif generation_mode == "标准质量":
                                            st.info(f"💡 {st.session_state.get('mode_desc', 'GIF')}：平衡质量和速度，采用中等分辨率({width}×{height})和智能帧数控制。")
                                        else:
                                            st.info(f"💡 {st.session_state.get('mode_desc', 'GIF')}：最高质量输出，采用高分辨率({width}×{height})，适合最终发布使用。")

                                        st.session_state.gif_generating = False

                                    elif st.session_state.get('gif_generating', True) and not images:
                                        progress_bar.empty()
                                        status_text.empty()
                                        st.error("❌ GIF生成失败，所有帧未能成功生成或被取消。")
                                        st.session_state.gif_generating = False
                                    else:
                                        if not st.session_state.get('gif_generating', True):
                                            progress_bar.empty()
                                            status_text.empty()
                                            st.warning("⚠️ GIF生成已取消")
                                        st.session_state.gif_generating = False

                                except ImportError:
                                    st.error("❌ 缺少必要的依赖库。请安装：pip install kaleido pillow")
                                    st.session_state.gif_generating = False
                                except Exception as gif_error:
                                    st.error(f"❌ GIF生成失败: {str(gif_error)}")
                                    st.info("💡 建议：如果GIF生成失败，可以选择HTML格式导出。")
                                    st.session_state.gif_generating = False

                    except Exception as e:
                        st.error(f"❌ 图表生成错误: {str(e)}")
            else:
                st.info("📋 请选择要可视化的数据列")
    else:
        st.info("📁 请先上传数据文件")
    
    with tab2:
        # 新的自定义轴配置功能
        st.markdown("### 🛠️ 自定义X轴和Y轴配置")
        st.markdown("在这里您可以自由选择X轴和Y轴数据列，并自定义图表样式。")

        if current_data is not None:
            # 创建自定义轴配置区域
            custom_config = UIComponents.create_custom_axis_section(current_data)

            if custom_config['x_column'] and custom_config['x_column'] is not None and custom_config['y_columns']:
                try:
                    # 创建图表生成器
                    chart_generator = ChartGenerator()

                    # 生成自定义轴图表
                    custom_fig = chart_generator.create_custom_axis_chart(current_data, custom_config)

                    # 显示图表
                    st.plotly_chart(custom_fig, use_container_width=True)

                    # 添加导出功能
                    st.markdown("---")
                    st.subheader("📥 导出自定义图表")

                    export_col1, export_col2 = st.columns(2)

                    with export_col1:
                        if st.button("📥 导出为HTML", type="primary", key="custom_html"):
                            html_string = custom_fig.to_html(
                                include_plotlyjs='cdn',
                                config={'displayModeBar': True, 'staticPlot': False}
                            )

                            st.download_button(
                                label="📥 下载HTML文件",
                                data=html_string,
                                file_name="custom_chart.html",
                                mime="text/html",
                                key="download_custom_html"
                            )

                            st.success("✅ HTML文件生成成功！")

                    with export_col2:
                        if st.button("📊 导出为PNG", key="custom_png"):
                            try:
                                img_bytes = custom_fig.to_image(format="png", width=1200, height=800)
                                st.download_button(
                                    label="📥 下载PNG文件",
                                    data=img_bytes,
                                    file_name="custom_chart.png",
                                    mime="image/png",
                                    key="download_custom_png"
                                )
                                st.success("✅ PNG文件生成成功！")
                            except Exception as e:
                                st.error(f"PNG导出失败: {str(e)}")
                                st.info("💡 提示：PNG导出需要安装kaleido库，请运行: pip install kaleido")

                except Exception as e:
                    st.error(f"❌ 自定义图表生成错误: {str(e)}")
            else:
                st.info("📊 请选择X轴和Y轴数据列来生成自定义图表")
        else:
            st.info("📊 请先上传数据文件")
    
    # Chat功能区域
    st.markdown("---")
    
    if current_data is not None:
        user_input, send_button, clear_button = UIComponents.create_chat_section()
        
        # 处理聊天交互
        if clear_button:
            SessionManager.clear_chat_history()
            st.rerun()
        
        if send_button and user_input.strip():
            # 增加input_key以清空输入框
            st.session_state.input_key += 1
            if not api_config['api_key']:
                st.error("❌ 请在侧边栏配置API密钥")
            else:
                try:
                    # 创建AI聊天实例
                    chat_processor = ChatProcessor()
                    
                    # 获取聊天历史
                    chat_history = st.session_state.get('chat_history', [])
                    
                    # 处理聊天请求
                    chat_response = chat_processor.process_chat_input(
                        user_input=user_input,
                        data=current_data,
                        deepseek_api_key=api_config['api_key'],
                        chat_history=chat_history
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
