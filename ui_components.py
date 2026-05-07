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
                "跳过数据行数（表头后）",
                min_value=0,
                max_value=50,
                value=0,
                help="在表头确定后，从数据区开头额外跳过的行数"
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", len(data))
        with col2:
            st.metric("总列数", len(data.columns))
        with col3:
            st.metric("缺失值", data.isnull().sum().sum())
        
        # 数据预览 - 不使用expander，避免嵌套
        st.markdown("### 📊 数据预览")
        preview_rows = st.slider("预览行数", 5, min(50, len(data)), 10)
        st.dataframe(data.head(preview_rows), use_container_width=True)
        
        # 数据清洗选项 - 不使用expander，避免嵌套
        st.markdown("### 🧹 数据清洗选项")
        col1, col2 = st.columns(2)
        
        with col1:
            remove_duplicates = st.checkbox("移除重复行", value=False)
            fill_na_method = st.selectbox(
                "缺失值处理",
                ["不处理", "删除含缺失值的行", "前向填充", "后向填充", "均值填充", "线性插值填充", "向前向后填充"],
                help="均值填充：数值列用均值填充，字符串列用向后填充（无法填充时用'未知'）；线性插值填充：数值列用线性插值，字符串列用前向填充；向前向后填充：先前向填充再后向填充。系统会自动将空格字符转换为真正的缺失值进行处理。"
            )
        
        with col2:
            # 获取所有列名
            all_columns = list(data.columns)
            auto_numeric_columns = UIComponents._detect_numeric_columns(data)
            
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

        # 数据筛选：按 FLIGHT_PHASE 列（仅当存在该列时显示）
        if auto_numeric_columns:
            st.caption(f"已自动识别 {len(auto_numeric_columns)} 列可转换为数值")

        selected_phases = None
        if 'FLIGHT_PHASE' in data.columns:
            st.markdown("### ✈️ 按飞行阶段筛选")
            try:
                phase_series = data['FLIGHT_PHASE'].dropna()
                unique_phases = sorted(phase_series.astype(str).unique().tolist())
            except Exception:
                # 回退方案，确保即使数据类型异常也能正常显示
                unique_phases = sorted(pd.Series(data['FLIGHT_PHASE']).dropna().astype(str).unique().tolist())
            selected_phases = st.multiselect(
                "选择要保留的FLIGHT_PHASE值",
                unique_phases,
                default=unique_phases,
                help="支持多选。若全部不选则不进行筛选；如果上传的CSV没有FLIGHT_PHASE列，此功能不显示。"
            )
        
        # 应用数据清洗
        cleaned_data = UIComponents._apply_data_cleaning(
            data, remove_duplicates, fill_na_method, convert_hash, convert_numeric
        )
        
        # 应用按 FLIGHT_PHASE 的数据筛选（在清洗完成后进行）
        if selected_phases is not None and len(selected_phases) > 0 and 'FLIGHT_PHASE' in cleaned_data.columns:
            try:
                cleaned_data = cleaned_data[cleaned_data['FLIGHT_PHASE'].astype(str).isin(selected_phases)]
            except Exception:
                # 出现异常时不影响整体流程
                st.warning("⚠️ 应用FLIGHT_PHASE筛选时出现问题，已跳过筛选。")
        
        return cleaned_data
    
    @staticmethod
    def create_column_selection_section(data: pd.DataFrame) -> Tuple[List[str], str, Optional[int]]:
        """
        创建列选择区域
        
        Args:
            data: 数据框
            
        Returns:
            tuple: (选中的列列表, 图表类型, 动画帧数)
        """
        st.subheader("📈 图表配置")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 数值列筛选
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_columns:
                st.error("❌ 没有找到数值类型的列，无法生成图表")
                return [], 'line', None
            
            selected_columns = st.multiselect(
                "选择要可视化的列",
                numeric_columns,
                default=[],
                help="选择要在图表中显示的数值列（建议不超过15列以保证显示效果）"
            )
        
        with col2:
            chart_type = st.selectbox(
                "图表类型",
                ['line', 'bar', 'scatter', 'animated'],
                format_func=lambda x: {'line': '📈 折线图', 'bar': '📊 柱状图', 'scatter': '🔵 散点图', 'animated': '🎬 动态图表'}[x]
            )
            
            # 动态图表的帧数选择
            animation_frames = None
            if chart_type == 'animated':
                animation_frames = st.selectbox(
                    "动画帧数设置",
                    [50, 100, 200, 500, 1000, "全部数据"],
                    index=1,  # 默认选择100
                    help="💡 提示：选择'全部数据'将播放每一个数据点，适合小数据集。大数据集建议选择较少帧数以保证播放流畅。"
                )
                # 如果选择"全部数据"，则设置为None，让程序自动使用所有数据点
                if animation_frames == "全部数据":
                    animation_frames = None
        
        return selected_columns, chart_type, animation_frames
    
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
    def create_custom_axis_section(data: pd.DataFrame) -> Dict[str, any]:
        """
        创建自定义X轴和Y轴配置区域
        
        Args:
            data: 数据框
            
        Returns:
            dict: 自定义轴配置结果
        """
        st.subheader("🎯 自定义轴配置")
        
        # 获取所有列（包括数值和非数值列）
        all_columns = data.columns.tolist()
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 X轴配置**")
            
            # X轴数据列选择
            x_column = st.selectbox(
                "选择X轴数据列",
                [None] + all_columns,  # 添加None选项作为默认
                index=0,  # 默认选择None
                format_func=lambda x: "请选择..." if x is None else str(x),
                help="选择作为X轴的数据列（可以是数值或文本列）"
            )
            
            # X轴标题
            x_title = st.text_input(
                "X轴标题",
                value=x_column if x_column else "",
                help="自定义X轴显示标题"
            )
            
            # X轴范围设置（仅对数值列）
            x_range_auto = True
            x_range = None
            if x_column and x_column in numeric_columns:
                x_range_auto = st.checkbox("自动X轴范围", value=True)
                if not x_range_auto:
                    x_min = data[x_column].min()
                    x_max = data[x_column].max()
                    x_range = st.slider(
                        "X轴范围",
                        min_value=float(x_min),
                        max_value=float(x_max),
                        value=(float(x_min), float(x_max)),
                        help="设置X轴显示范围"
                    )
        
        with col2:
            st.markdown("**📈 Y轴配置**")
            
            # Y轴数据列选择（多选）
            y_columns = st.multiselect(
                "选择Y轴数据列",
                numeric_columns,
                default=[],  # 默认为空
                help="选择作为Y轴的数值列（可多选）"
            )
            
            # Y轴标题
            y_title = st.text_input(
                "Y轴标题",
                value="数值" if y_columns else "",
                help="自定义Y轴显示标题"
            )
            
            # Y轴范围设置
            y_range_auto = True
            y_range = None
            if y_columns:
                y_range_auto = st.checkbox("自动Y轴范围", value=True)
                if not y_range_auto:
                    y_data = data[y_columns].select_dtypes(include=['number'])
                    if not y_data.empty:
                        y_min = y_data.min().min()
                        y_max = y_data.max().max()
                        y_range = st.slider(
                            "Y轴范围",
                            min_value=float(y_min),
                            max_value=float(y_max),
                            value=(float(y_min), float(y_max)),
                            help="设置Y轴显示范围"
                        )
        
        # 图表样式配置
        st.markdown("**🎨 图表样式**")
        style_col1, style_col2, style_col3 = st.columns(3)
        
        with style_col1:
            chart_type = st.selectbox(
                "图表类型",
                ['line', 'bar', 'scatter', 'area', 'pie', 'donut', 'radar'],
                format_func=lambda x: {
                    'line': '📈 折线图',
                    'bar': '📊 柱状图', 
                    'scatter': '🔵 散点图',
                    'area': '📊 面积图',
                    'pie': '🥧 饼图',
                    'donut': '🍩 环形图',
                    'radar': '🎯 雷达图'
                }[x],
                help="选择图表显示类型"
            )
        
        with style_col2:
            # 颜色主题选项映射
            color_theme_options = {
                '默认主题': 'plotly',
                '翠绿渐变': 'viridis', 
                '紫红渐变': 'plasma',
                '火焰渐变': 'inferno',
                '岩浆渐变': 'magma',
                '蓝绿渐变': 'cividis'
            }
            
            color_theme_display = st.selectbox(
                "颜色主题",
                list(color_theme_options.keys()),
                help="选择图表颜色主题"
            )
            
            # 获取对应的英文值
            color_theme = color_theme_options[color_theme_display]
        
        with style_col3:
            show_grid = st.checkbox("显示网格线", value=True)
        
        return {
            'x_column': x_column,
            'x_title': x_title,
            'x_range_auto': x_range_auto,
            'x_range': x_range,
            'y_columns': y_columns,
            'y_title': y_title,
            'y_range_auto': y_range_auto,
            'y_range': y_range,
            'chart_type': chart_type,
            'color_theme': color_theme,
            'show_grid': show_grid
        }
    
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
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == "user":
                    st.markdown(f"**🙋 用户:** {message['content']}")
                else:
                    st.markdown(f"**🤖 助手:** {message['content']}")
                st.markdown("---")
        
        # 输入区域
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            # 使用session_state来控制输入框的值
            if 'input_key' not in st.session_state:
                st.session_state.input_key = 0
            
            user_input = st.text_input(
                "输入您的问题",
                placeholder="例如：分析一下数据的趋势...",
                key=f"chat_input_{st.session_state.input_key}"
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
        from ai_chat import ChatProcessor
        
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
            
            # API连接验证部分
            col1, col2 = st.columns([1, 1])
            
            with col1:
                test_button = st.button(
                    "🔍 测试连接", 
                    use_container_width=True,
                    disabled=not api_key or api_key.strip() == ""
                )
            
            with col2:
                if st.button("🗑️ 清除状态", use_container_width=True):
                    if 'api_test_result' in st.session_state:
                        del st.session_state.api_test_result
            
            # 执行API测试
            if test_button and api_key:
                with st.spinner("正在测试API连接..."):
                    test_result = ChatProcessor.test_api_key(api_key)
                    st.session_state.api_test_result = test_result
            
            # 显示测试结果
            if 'api_test_result' in st.session_state:
                result = st.session_state.api_test_result
                
                if result['success']:
                    st.success(f"{result['message']}\n响应时间: {result['response_time']}秒")
                else:
                    st.error(result['message'])
                    if result['response_time'] > 0:
                        st.caption(f"响应时间: {result['response_time']}秒")
            
            # 如果有API密钥但未测试，显示提示
            elif api_key and api_key.strip() != "":
                st.warning("💡 建议点击'测试连接'验证API密钥是否有效")
            
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
            cleaned_data[col] = cleaned_data[col].replace('None', pd.NA)
            cleaned_data[col] = cleaned_data[col].replace('none', pd.NA)
            cleaned_data[col] = cleaned_data[col].replace('null', pd.NA)
            cleaned_data[col] = cleaned_data[col].replace('NULL', pd.NA)
        
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
        elif fill_na_method == "线性插值填充":
            # 数值列用线性插值填充
            numeric_cols = cleaned_data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                cleaned_data[col] = cleaned_data[col].interpolate(method='linear')
            # 字符串列用前向填充
            string_cols = cleaned_data.select_dtypes(include=['object', 'string']).columns
            if not string_cols.empty:
                cleaned_data[string_cols] = cleaned_data[string_cols].ffill()
                # 如果前向填充后仍有缺失值，用'未知'填充
                cleaned_data[string_cols] = cleaned_data[string_cols].fillna('未知')
        elif fill_na_method == "向前向后填充":
            # 先前向填充，再后向填充
            cleaned_data = cleaned_data.ffill().bfill()
            # 如果仍有缺失值，数值列用0填充，字符串列用'未知'填充
            numeric_cols = cleaned_data.select_dtypes(include=['number']).columns
            string_cols = cleaned_data.select_dtypes(include=['object', 'string']).columns
            if not numeric_cols.empty:
                cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(0)
            if not string_cols.empty:
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
        auto_numeric_columns = UIComponents._detect_numeric_columns(cleaned_data)
        for col in cleaned_data.columns:
            if col in auto_numeric_columns or col in convert_hash:
                continue

            if cleaned_data[col].dtype not in ['object', 'string']:
                continue

            probe = cleaned_data[col].dropna().astype(str).str.strip()
            probe = probe.replace(['', 'nan', 'None', 'none', 'null', 'NULL'], pd.NA).dropna()
            if probe.empty:
                continue

            probe_numeric = pd.to_numeric(probe.str.replace(',', '', regex=False), errors='coerce')
            if probe_numeric.notna().mean() >= 0.8:
                auto_numeric_columns.append(col)

        columns_to_convert = list(dict.fromkeys(convert_numeric + auto_numeric_columns))

        for col in columns_to_convert:
            try:
                cleaned_data[col] = pd.to_numeric(
                    cleaned_data[col].astype(str).str.replace(',', '', regex=False),
                    errors='coerce'
                )
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
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 使用字典格式保持与ai_chat.py的一致性
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
    
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
