# -*- coding: utf-8 -*-
"""
UI组件模块 - 负责Streamlit界面组件的创建和管理
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from config import APP_CONFIG, COPYRIGHT_INFO


class UIComponents:
    """
    UI组件类，负责创建和管理各种界面组件
    """
    
    @staticmethod
    def setup_page_config():
        """
        设置页面配置
        """
        st.set_page_config(
            page_title=APP_CONFIG['page_title'],
            page_icon=APP_CONFIG['page_icon'],
            layout=APP_CONFIG['layout']
        )
    
    @staticmethod
    def create_header():
        """
        创建页面标题
        """
        st.title(APP_CONFIG['page_title'])
        st.markdown("---")
    
    @staticmethod
    def create_file_upload_section() -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        创建文件上传区域
        
        Returns:
            tuple: (上传的文件对象, 文件处理参数)
        """
        st.subheader("📁 数据文件上传")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "选择Excel或CSV文件",
                type=['xlsx', 'xls', 'csv'],
                help="支持Excel (.xlsx, .xls) 和 CSV (.csv) 格式"
            )
        
        with col2:
            st.markdown("**文件处理选项**")
            header_row = st.number_input(
                "标题行位置",
                min_value=0,
                max_value=10,
                value=0,
                help="指定哪一行作为列标题（从0开始计数）"
            )
            
            skip_rows = st.number_input(
                "跳过行数",
                min_value=0,
                max_value=50,
                value=0,
                help="从文件开头跳过的行数"
            )
        
        file_params = {
            'header_row': header_row,
            'skip_rows': skip_rows
        }
        
        return uploaded_file, file_params
    
    @staticmethod
    def create_data_preview_section(data: pd.DataFrame) -> pd.DataFrame:
        """
        创建数据预览和清洗区域
        
        Args:
            data: 原始数据框
            
        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        st.subheader("🔍 数据预览与清洗")
        
        # 数据基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总行数", len(data))
        with col2:
            st.metric("总列数", len(data.columns))
        with col3:
            st.metric("缺失值", data.isnull().sum().sum())
        with col4:
            st.metric("数据类型", len(data.dtypes.unique()))
        
        # 数据预览
        with st.expander("📊 数据预览", expanded=True):
            preview_rows = st.slider("预览行数", 5, min(50, len(data)), 10)
            st.dataframe(data.head(preview_rows), use_container_width=True)
        
        # 数据清洗选项
        with st.expander("🧹 数据清洗选项"):
            col1, col2 = st.columns(2)
            
            with col1:
                remove_duplicates = st.checkbox("移除重复行", value=False)
                fill_na_method = st.selectbox(
                    "缺失值处理",
                    ["不处理", "删除含缺失值的行", "前向填充", "后向填充", "均值填充"],
                    help="均值填充：数值列用均值填充，字符串列用向后填充（无法填充时用'未知'）。系统会自动将空格字符转换为真正的缺失值进行处理。"
                )
            
            with col2:
                # 获取所有列名
                all_columns = list(data.columns)
                
                # 转换为哈希值的列
                convert_hash = st.multiselect(
                    "转换为哈希值的列",
                    all_columns,
                    default=[],
                    help="选择要将字符串转换为哈希值的列（便于在图表中显示）"
                )
                
                # 自动检测可能的数值列
                convert_numeric = st.multiselect(
                    "转换为数值格式的列",
                    all_columns,
                    default=[],
                    help="选择要转换为数值格式的列"
                )
        
        # 应用数据清洗
        cleaned_data = UIComponents._apply_data_cleaning(
            data, remove_duplicates, fill_na_method, convert_hash, convert_numeric
        )
        
        return cleaned_data
    
    @staticmethod
    def create_column_selection_section(data: pd.DataFrame) -> Tuple[List[str], str]:
        """
        创建列选择区域
        
        Args:
            data: 数据框
            
        Returns:
            tuple: (选中的列列表, 图表类型)
        """
        st.subheader("📈 图表配置")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 数值列筛选
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_columns:
                st.error("❌ 没有找到数值类型的列，无法生成图表")
                return [], 'line'
            
            selected_columns = st.multiselect(
                "选择要可视化的列",
                numeric_columns,
                default=numeric_columns[:min(10, len(numeric_columns))],
                help="选择要在图表中显示的数值列（建议不超过15列以保证显示效果）"
            )
        
        with col2:
            chart_type = st.selectbox(
                "图表类型",
                ['line', 'bar', 'scatter'],
                format_func=lambda x: {'line': '📈 折线图', 'bar': '📊 柱状图', 'scatter': '🔵 散点图'}[x]
            )
        
        return selected_columns, chart_type
    
    @staticmethod
    def create_chart_layout_section() -> str:
        """
        创建图表布局选择区域
        
        Returns:
            str: 选择的布局类型
        """
        st.subheader("🎨 图表布局")
        
        layout_options = {
            'single': '📈 单轴图表',
            'dual': '📊 双轴图表',
            'triple': '📉 三轴图表',
            'subplot': '🔲 子图表',
            'compact': '📋 紧凑子图'
        }
        
        layout_type = st.selectbox(
            "选择图表布局",
            list(layout_options.keys()),
            format_func=lambda x: layout_options[x],
            help="选择图表的显示布局方式"
        )
        
        return layout_type
    
    @staticmethod
    def create_axis_assignment_section(selected_columns: List[str], layout_type: str) -> Dict[str, List[str]]:
        """
        创建轴分配区域
        
        Args:
            selected_columns: 选中的列
            layout_type: 布局类型
            
        Returns:
            dict: 轴分配结果
        """
        if layout_type in ['dual', 'triple']:
            st.subheader("⚖️ 轴分配")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                primary_cols = st.multiselect(
                    "主轴 (左侧)",
                    selected_columns,
                    default=selected_columns[:len(selected_columns)//2] if selected_columns else [],
                    help="显示在左侧Y轴的数据列"
                )
            
            with col2:
                remaining_cols = [col for col in selected_columns if col not in primary_cols]
                secondary_cols = st.multiselect(
                    "副轴 (右侧)",
                    remaining_cols,
                    default=remaining_cols[:len(remaining_cols)//2] if remaining_cols else [],
                    help="显示在右侧Y轴的数据列"
                )
            
            with col3:
                if layout_type == 'triple':
                    remaining_cols2 = [col for col in remaining_cols if col not in secondary_cols]
                    third_cols = st.multiselect(
                        "第三轴",
                        remaining_cols2,
                        default=remaining_cols2,
                        help="显示在第三Y轴的数据列"
                    )
                else:
                    third_cols = []
            
            return {
                'primary': primary_cols,
                'secondary': secondary_cols,
                'third': third_cols
            }
        
        return {'primary': selected_columns, 'secondary': [], 'third': []}
    
    @staticmethod
    def create_chat_section() -> Tuple[str, bool, bool]:
        """
        创建聊天区域
        
        Returns:
            tuple: (用户输入, 是否发送, 是否清空)
        """
        st.subheader("💬 Chat with Excel")
        
        # 显示聊天历史
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 聊天历史容器
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f"**🙋 用户:** {message}")
                else:
                    st.markdown(f"**🤖 助手:** {message}")
                st.markdown("---")
        
        # 输入区域
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            user_input = st.text_input(
                "输入您的问题",
                placeholder="例如：分析一下数据的趋势...",
                key="chat_input"
            )
        
        with col2:
            send_button = st.button("📤 发送", use_container_width=True)
        
        with col3:
            clear_button = st.button("🗑️ 清空", use_container_width=True)
        
        return user_input, send_button, clear_button
    
    @staticmethod
    def create_api_config_section() -> Dict[str, str]:
        """
        创建API配置区域
        
        Returns:
            dict: API配置信息
        """
        with st.sidebar:
            st.subheader("🔧 AI配置")
            
            # 只保留DeepSeek API选项
            model_provider = "DeepSeek API"
            st.info("当前使用: DeepSeek API")
            
            api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                help="请输入您的DeepSeek API密钥"
            )
            
            return {
                'provider': model_provider,
                'api_key': api_key
            }
    
    @staticmethod
    def create_sidebar_info():
        """
        创建侧边栏信息
        """
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📊 功能特点")
            st.markdown("""
            - 🔄 智能文件读取
            - 📈 多种图表类型
            - 🎨 灵活布局选择
            - 💬 AI数据分析
            - 🔍 交互式缩放
            """)
            
            st.markdown("---")
            st.markdown(f"**{COPYRIGHT_INFO['text']}**")
            st.markdown(f"📧 问题反馈: {COPYRIGHT_INFO['email']}")
    
    @staticmethod
    def _detect_date_columns(data: pd.DataFrame) -> List[str]:
        """
        检测可能的日期列
        
        Args:
            data: 数据框
            
        Returns:
            list: 可能的日期列名列表
        """
        date_columns = []
        for col in data.columns:
            # 检测所有列类型，不仅仅是object类型
            if data[col].dtype == 'object' or 'datetime' in str(data[col].dtype):
                # 简单的日期格式检测
                sample_values = data[col].dropna().head(10).astype(str)
                date_patterns = ['-', '/', '年', '月', '日', 'time', 'date', 'Time', 'Date', 'TIME', 'DATE']
                if any(pattern in ' '.join(sample_values) for pattern in date_patterns) or 'datetime' in str(data[col].dtype):
                    date_columns.append(col)
            # 检测列名中包含时间相关关键词的列
            elif any(keyword in col.lower() for keyword in ['time', 'date', '时间', '日期', 'datetime', 'timestamp']):
                date_columns.append(col)
        return date_columns
    
    @staticmethod
    def _detect_numeric_columns(data: pd.DataFrame) -> List[str]:
        """
        检测可能的数值列
        
        Args:
            data: 数据框
            
        Returns:
            list: 可能的数值列名列表
        """
        numeric_columns = []
        for col in data.columns:
            # 检测所有非数值类型的列，看是否可以转换为数值
            if data[col].dtype == 'object':
                # 尝试转换为数值
                try:
                    pd.to_numeric(data[col].dropna().head(10), errors='raise')
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    pass
            # 已经是数值类型但可能需要进一步处理的列
            elif data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # 如果列中有缺失值或者数据类型不一致，也加入候选列表
                if data[col].isnull().any() or data[col].dtype == 'object':
                    numeric_columns.append(col)
        return numeric_columns
    
    @staticmethod
    def _apply_data_cleaning(data: pd.DataFrame, remove_duplicates: bool, 
                           fill_na_method: str, convert_hash: List[str], 
                           convert_numeric: List[str]) -> pd.DataFrame:
        """
        应用数据清洗操作
        
        Args:
            data: 原始数据框
            remove_duplicates: 是否移除重复行
            fill_na_method: 缺失值处理方法
            convert_hash: 要转换为哈希值的列
            convert_numeric: 要转换为数值的列
            
        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        cleaned_data = data.copy()
        
        # 预处理：将空格字符转换为NaN
        # 这样可以确保空格不会被误认为有效数据
        for col in cleaned_data.select_dtypes(include=['object', 'string']).columns:
            # 将只包含空格的字符串替换为NaN
            cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
            cleaned_data[col] = cleaned_data[col].replace('', pd.NA)
            cleaned_data[col] = cleaned_data[col].replace('nan', pd.NA)
        
        # 移除重复行
        if remove_duplicates:
            cleaned_data = cleaned_data.drop_duplicates()
        
        # 处理缺失值
        if fill_na_method == "删除含缺失值的行":
            cleaned_data = cleaned_data.dropna()
        elif fill_na_method == "前向填充":
            cleaned_data = cleaned_data.ffill()
        elif fill_na_method == "后向填充":
            cleaned_data = cleaned_data.bfill()
        elif fill_na_method == "均值填充":
            # 数值列用均值填充
            numeric_cols = cleaned_data.select_dtypes(include=['number']).columns
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                cleaned_data[numeric_cols].mean()
            )
            # 字符串列用向后填充
            string_cols = cleaned_data.select_dtypes(include=['object', 'string']).columns
            if not string_cols.empty:
                cleaned_data[string_cols] = cleaned_data[string_cols].bfill()
                # 如果向后填充后仍有缺失值，用'未知'填充
                cleaned_data[string_cols] = cleaned_data[string_cols].fillna('未知')
        
        # 转换为哈希值列
        if 'string_mappings' not in st.session_state:
            st.session_state.string_mappings = {}
            
        for col in convert_hash:
            try:
                # 创建字符串到哈希值的映射
                string_to_hash = {}
                hash_values = []
                
                for value in cleaned_data[col].astype(str):
                    if pd.notna(value) and value != 'nan':
                        if value not in string_to_hash:
                            string_to_hash[value] = hash(value) % 1000000
                        hash_values.append(string_to_hash[value])
                    else:
                        hash_values.append(0)
                
                # 保存映射关系到session_state
                st.session_state.string_mappings[col] = string_to_hash
                
                # 将字符串转换为哈希值，便于在图表中显示
                cleaned_data[col] = hash_values
            except Exception:
                st.warning(f"无法将列 '{col}' 转换为哈希值")
        
        # 转换数值列
        for col in convert_numeric:
            try:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            except Exception:
                st.warning(f"无法将列 '{col}' 转换为数值格式")
        
        return cleaned_data


class SessionManager:
    """
    会话管理器，负责管理Streamlit会话状态
    """
    
    @staticmethod
    def initialize_session_state():
        """
        初始化会话状态
        """
        default_states = {
            'chat_history': [],
            'data_loaded': False,
            'current_data': None,
            'chart_config': {},
            'zoom_state': {
                'primary_auto': True,
                'secondary_auto': True,
                'third_auto': True,
                'primary_range': [0, 100],
                'secondary_range': [0, 100],
                'third_range': [0, 100]
            }
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def update_chat_history(user_message: str, assistant_message: str):
        """
        更新聊天历史
        
        Args:
            user_message: 用户消息
            assistant_message: 助手回复
        """
        st.session_state.chat_history.append(("user", user_message))
        st.session_state.chat_history.append(("assistant", assistant_message))
    
    @staticmethod
    def clear_chat_history():
        """
        清空聊天历史
        """
        st.session_state.chat_history = []
    
    @staticmethod
    def save_data(data: pd.DataFrame):
        """
        保存数据到会话状态
        
        Args:
            data: 要保存的数据框
        """
        st.session_state.current_data = data
        st.session_state.data_loaded = True
    
    @staticmethod
    def get_current_data() -> Optional[pd.DataFrame]:
        """
        获取当前数据
        
        Returns:
            pd.DataFrame or None: 当前数据框
        """
        return st.session_state.get('current_data', None)