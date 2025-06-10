# -*- coding: utf-8 -*-
"""
模块化的Excel数据可视化分析工具
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

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
                        selected_columns, chart_type = UIComponents.create_column_selection_section(cleaned_data)
                        
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
                                'axis_assignment': axis_assignment
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
                            config['chart_type']
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
