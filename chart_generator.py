# -*- coding: utf-8 -*-
"""
图表生成模块 - 负责数据可视化和图表生成
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Any, Optional
from config import COLOR_SCHEMES, CHART_CONFIG


class ChartGenerator:
    """
    图表生成器类，负责创建各种类型的图表
    """
    
    def __init__(self, color_scheme: str = 'default'):
        self.colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['default'])
        self.chart_config = CHART_CONFIG
    
    def create_single_axis_chart(self, data: pd.DataFrame, selected_columns: List[str], 
                                chart_type: str = 'line') -> go.Figure:
        """
        创建单轴图表
        
        Args:
            data: 数据框
            selected_columns: 选中的列
            chart_type: 图表类型
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        fig = go.Figure()
        
        for i, col in enumerate(selected_columns):
            color = self.colors[i % len(self.colors)]
            
            # 检查是否为哈希值列（数值范围在0-999999之间且原始数据包含字符串）
            is_hash_column = self._is_hash_column(data, col)
            
            if chart_type == 'line':
                if is_hash_column:
                    # 获取原始字符串数据用于悬停显示
                    original_strings = self._get_original_strings(data, col)
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=pd.to_numeric(data[col], errors='coerce'),
                        mode='lines',
                        name=col,
                        line=dict(color=color, width=2),
                        customdata=original_strings,
                        hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=pd.to_numeric(data[col], errors='coerce'),
                        mode='lines',
                        name=col,
                        line=dict(color=color, width=2)
                    ))
            elif chart_type == 'bar':
                if is_hash_column:
                    original_strings = self._get_original_strings(data, col)
                    fig.add_trace(go.Bar(
                        x=data.index,
                        y=pd.to_numeric(data[col], errors='coerce'),
                        name=col,
                        marker_color=color,
                        customdata=original_strings,
                        hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=data.index,
                        y=pd.to_numeric(data[col], errors='coerce'),
                        name=col,
                        marker_color=color
                    ))
            elif chart_type == 'scatter':
                if is_hash_column:
                    original_strings = self._get_original_strings(data, col)
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=pd.to_numeric(data[col], errors='coerce'),
                        mode='markers',
                        name=col,
                        marker=dict(color=color, size=6),
                        customdata=original_strings,
                        hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=pd.to_numeric(data[col], errors='coerce'),
                        mode='markers',
                        name=col,
                        marker=dict(color=color, size=6)
                    ))
        
        self._apply_layout_config(fig, data, selected_columns)
        return fig
    
    def create_dual_axis_chart(self, data: pd.DataFrame, primary_cols: List[str], 
                              secondary_cols: List[str]) -> go.Figure:
        """
        创建双轴图表
        
        Args:
            data: 数据框
            primary_cols: 主轴列
            secondary_cols: 副轴列
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        fig = go.Figure()
        
        # 添加主轴数据
        for i, col in enumerate(primary_cols):
            color = self.colors[i % len(self.colors)]
            is_hash_column = self._is_hash_column(data, col)
            
            if is_hash_column:
                original_strings = self._get_original_strings(data, col)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (主轴)",
                    line=dict(color=color, width=2),
                    yaxis='y',
                    customdata=original_strings,
                    hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (主轴)",
                    line=dict(color=color, width=2),
                    yaxis='y'
                ))
        
        # 添加副轴数据
        for i, col in enumerate(secondary_cols):
            color = self.colors[(len(primary_cols) + i) % len(self.colors)]
            is_hash_column = self._is_hash_column(data, col)
            
            if is_hash_column:
                original_strings = self._get_original_strings(data, col)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (副轴)",
                    line=dict(color=color, width=2),
                    yaxis='y2',
                    customdata=original_strings,
                    hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (副轴)",
                    line=dict(color=color, width=2),
                    yaxis='y2'
                ))
        
        # 配置Y轴刻度标签
        yaxis_config = dict(**self.chart_config['grid_config'], title="主轴")
        yaxis2_config = dict(**self.chart_config['grid_config'], overlaying='y', side='right', title="副轴")
        
        # 为哈希值列设置自定义Y轴刻度
        if primary_cols:
            primary_hash_cols = [col for col in primary_cols if self._is_hash_column(data, col)]
            if primary_hash_cols:
                # 获取主轴哈希值列的唯一值和对应的原始字符串
                col = primary_hash_cols[0]  # 使用第一个哈希值列
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    yaxis_config.update({
                        'tickmode': 'array',
                        'tickvals': unique_hashes,
                        'ticktext': [hash_to_string.get(h, str(h)) for h in unique_hashes]
                    })
        
        if secondary_cols:
            secondary_hash_cols = [col for col in secondary_cols if self._is_hash_column(data, col)]
            if secondary_hash_cols:
                # 获取副轴哈希值列的唯一值和对应的原始字符串
                col = secondary_hash_cols[0]  # 使用第一个哈希值列
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    yaxis2_config.update({
                        'tickmode': 'array',
                        'tickvals': unique_hashes,
                        'ticktext': [hash_to_string.get(h, str(h)) for h in unique_hashes]
                    })
        
        # 配置双轴布局
        fig.update_layout(
            showlegend=True,
            width=self.chart_config['default_width'],
            height=self.chart_config['default_height'],
            hovermode='x',
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                font_size=12,
                align="left"
            ),
            xaxis=dict(
                **self.chart_config['grid_config'],
                tickmode='linear',
                dtick=300,
                tickangle=45
            ),
            yaxis=yaxis_config,
            yaxis2=yaxis2_config
        )
        
        fig.update_xaxes(rangeslider=self.chart_config['rangeslider_config'])
        return fig
    
    def create_triple_axis_chart(self, data: pd.DataFrame, primary_cols: List[str], 
                                secondary_cols: List[str], third_cols: List[str]) -> go.Figure:
        """
        创建三轴图表
        
        Args:
            data: 数据框
            primary_cols: 主轴列
            secondary_cols: 副轴列
            third_cols: 第三轴列
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        fig = go.Figure()
        
        # 添加主轴数据
        for i, col in enumerate(primary_cols):
            color = self.colors[i % len(self.colors)]
            is_hash_column = self._is_hash_column(data, col)
            
            if is_hash_column:
                original_strings = self._get_original_strings(data, col)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (主轴)",
                    line=dict(color=color, width=2),
                    yaxis='y',
                    customdata=original_strings,
                    hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (主轴)",
                    line=dict(color=color, width=2),
                    yaxis='y'
                ))
        
        # 添加副轴数据
        for i, col in enumerate(secondary_cols):
            color = self.colors[(len(primary_cols) + i) % len(self.colors)]
            is_hash_column = self._is_hash_column(data, col)
            
            if is_hash_column:
                original_strings = self._get_original_strings(data, col)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (副轴)",
                    line=dict(color=color, width=2),
                    yaxis='y2',
                    customdata=original_strings,
                    hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (副轴)",
                    line=dict(color=color, width=2),
                    yaxis='y2'
                ))
        
        # 添加第三轴数据
        for i, col in enumerate(third_cols):
            color = self.colors[(len(primary_cols) + len(secondary_cols) + i) % len(self.colors)]
            is_hash_column = self._is_hash_column(data, col)
            
            if is_hash_column:
                original_strings = self._get_original_strings(data, col)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (第三轴)",
                    line=dict(color=color, width=2),
                    yaxis='y3',
                    customdata=original_strings,
                    hovertemplate=f'{col}: %{{customdata}}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=pd.to_numeric(data[col], errors='coerce'),
                    mode='lines',
                    name=f"{col} (第三轴)",
                    line=dict(color=color, width=2),
                    yaxis='y3'
                ))
        
        # 配置Y轴刻度标签
        yaxis_config = dict(**self.chart_config['grid_config'], title="主轴", side='left')
        yaxis2_config = dict(**self.chart_config['grid_config'], overlaying='y', side='right', title="副轴")
        yaxis3_config = dict(
            **self.chart_config['grid_config'],
            overlaying='y',
            side='right',
            position=0.95,
            title=dict(text="第三轴", font=dict(color='red')),
            tickfont=dict(color='red')
        )
        
        # 为哈希值列设置自定义Y轴刻度
        if primary_cols:
            primary_hash_cols = [col for col in primary_cols if self._is_hash_column(data, col)]
            if primary_hash_cols:
                col = primary_hash_cols[0]
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    yaxis_config.update({
                        'tickmode': 'array',
                        'tickvals': unique_hashes,
                        'ticktext': [hash_to_string.get(h, str(h)) for h in unique_hashes]
                    })
        
        if secondary_cols:
            secondary_hash_cols = [col for col in secondary_cols if self._is_hash_column(data, col)]
            if secondary_hash_cols:
                col = secondary_hash_cols[0]
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    yaxis2_config.update({
                        'tickmode': 'array',
                        'tickvals': unique_hashes,
                        'ticktext': [hash_to_string.get(h, str(h)) for h in unique_hashes]
                    })
        
        if third_cols:
            third_hash_cols = [col for col in third_cols if self._is_hash_column(data, col)]
            if third_hash_cols:
                col = third_hash_cols[0]
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    yaxis3_config.update({
                        'tickmode': 'array',
                        'tickvals': unique_hashes,
                        'ticktext': [hash_to_string.get(h, str(h)) for h in unique_hashes]
                    })
        
        # 配置三轴布局
        fig.update_layout(
            showlegend=True,
            width=self.chart_config['default_width'],
            height=self.chart_config['default_height'],
            hovermode='x',
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                font_size=12,
                align="left"
            ),
            xaxis=dict(
                **self.chart_config['grid_config'],
                domain=[0.1, 0.9],  # 为第三轴留出空间
                tickmode='linear',
                dtick=300,
                tickangle=45
            ),
            yaxis=yaxis_config,
            yaxis2=yaxis2_config,
            yaxis3=yaxis3_config
        )
        
        fig.update_xaxes(rangeslider=self.chart_config['rangeslider_config'])
        return fig
    
    def create_subplot_charts(self, data: pd.DataFrame, selected_columns: List[str],
                             cols: int = 2, chart_type: str = 'line') -> go.Figure:
        """
        创建子图表
        
        Args:
            data: 数据框
            selected_columns: 选中的列
            cols: 列数
            chart_type: 图表类型
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        num_plots = len(selected_columns)
        rows = (num_plots + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=selected_columns,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            shared_xaxes=True
        )
        
        for i, col in enumerate(selected_columns):
            row = i // cols + 1
            col_pos = i % cols + 1
            color = self.colors[i % len(self.colors)]
            
            # 检查是否为哈希值列并获取原始字符串
            y_values = pd.to_numeric(data[col], errors='coerce')
            hover_text = None
            if self._is_hash_column(data, col):
                original_strings = self._get_original_strings(data, col)
                hover_text = [f"{col}: {orig}<br>索引: {idx}" for idx, orig in zip(data.index, original_strings)]
            
            if chart_type == 'line':
                trace = go.Scatter(
                    x=data.index,
                    y=y_values,
                    mode='lines',
                    name=col,
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertext=hover_text,
                    hovertemplate='%{hovertext}<extra></extra>' if hover_text else None
                )
            elif chart_type == 'bar':
                trace = go.Bar(
                    x=data.index,
                    y=y_values,
                    name=col,
                    marker_color=color,
                    showlegend=False,
                    hovertext=hover_text,
                    hovertemplate='%{hovertext}<extra></extra>' if hover_text else None
                )
            elif chart_type == 'scatter':
                trace = go.Scatter(
                    x=data.index,
                    y=y_values,
                    mode='markers',
                    name=col,
                    marker=dict(color=color, size=4),
                    showlegend=False,
                    hovertext=hover_text,
                    hovertemplate='%{hovertext}<extra></extra>' if hover_text else None
                )
            
            fig.add_trace(trace, row=row, col=col_pos)
            
            # 为哈希值列设置自定义Y轴刻度
            if self._is_hash_column(data, col):
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    fig.update_yaxes(
                        tickmode='array',
                        tickvals=unique_hashes,
                        ticktext=[hash_to_string.get(h, str(h)) for h in unique_hashes],
                        row=row,
                        col=col_pos
                    )
        
        # 只在最下方的子图显示X轴刻度
        for i in range(len(selected_columns)):
            row = i // cols + 1
            col_pos = i % cols + 1
            if row < rows:  # 不是最后一行的子图
                fig.update_xaxes(showticklabels=False, row=row, col=col_pos)
        
        fig.update_layout(
            height=300 * rows,
            width=self.chart_config['default_width'],
            showlegend=False
        )
        
        return fig
    
    def create_compact_subplot(self, data: pd.DataFrame, selected_columns: List[str]) -> go.Figure:
        """
        创建紧凑型子图表
        
        Args:
            data: 数据框
            selected_columns: 选中的列
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        num_plots = len(selected_columns)
        
        fig = make_subplots(
            rows=num_plots,
            cols=1,
            shared_xaxes=True,
            subplot_titles=selected_columns,
            vertical_spacing=0.02
        )
        
        for i, col in enumerate(selected_columns):
            color = self.colors[i % len(self.colors)]
            
            # 检查是否为哈希值列并获取原始字符串
            y_values = pd.to_numeric(data[col], errors='coerce')
            hover_text = None
            if self._is_hash_column(data, col):
                original_strings = self._get_original_strings(data, col)
                hover_text = [f"{col}: {orig}<br>索引: {idx}" for idx, orig in zip(data.index, original_strings)]
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=y_values,
                    mode='lines',
                    name=col,
                    line=dict(color=color, width=1.5),
                    showlegend=False,
                    hovertext=hover_text,
                    hovertemplate='%{hovertext}<extra></extra>' if hover_text else None
                ),
                row=i + 1,
                col=1
            )
            
            # 为哈希值列设置自定义Y轴刻度
            if self._is_hash_column(data, col):
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    fig.update_yaxes(
                        tickmode='array',
                        tickvals=unique_hashes,
                        ticktext=[hash_to_string.get(h, str(h)) for h in unique_hashes],
                        row=i + 1,
                        col=1
                    )
        
        fig.update_layout(
            height=150 * num_plots,
            width=self.chart_config['default_width'],
            showlegend=False
        )
        
        return fig
    
    def _apply_layout_config(self, fig: go.Figure, data: pd.DataFrame = None, columns: List[str] = None):
        """
        应用通用布局配置
        
        Args:
            fig: 图表对象
            data: 数据框（可选，用于配置哈希值列的Y轴刻度）
            columns: 列名列表（可选，用于配置哈希值列的Y轴刻度）
        """
        yaxis_config = dict(**self.chart_config['grid_config'])
        
        # 为哈希值列设置自定义Y轴刻度
        if data is not None and columns:
            hash_cols = [col for col in columns if self._is_hash_column(data, col)]
            if hash_cols:
                col = hash_cols[0]  # 使用第一个哈希值列
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    yaxis_config.update({
                        'tickmode': 'array',
                        'tickvals': unique_hashes,
                        'ticktext': [hash_to_string.get(h, str(h)) for h in unique_hashes]
                    })
        
        fig.update_layout(
            showlegend=True,
            width=self.chart_config['default_width'],
            height=self.chart_config['default_height'],
            hovermode='x',
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                font_size=12,
                align="left"
            ),
            xaxis=dict(
                **self.chart_config['grid_config'],
                tickmode='linear',
                dtick=300,
                tickangle=45
            ),
            yaxis=yaxis_config
        )
        
        fig.update_xaxes(rangeslider=self.chart_config['rangeslider_config'])
    
    def _is_hash_column(self, data: pd.DataFrame, col: str) -> bool:
        """
        检测列是否为哈希值列
        
        Args:
            data: 数据框
            col: 列名
            
        Returns:
            bool: 是否为哈希值列
        """
        try:
            # 检查列是否为数值类型且在哈希值范围内
            numeric_data = pd.to_numeric(data[col], errors='coerce')
            if numeric_data.isna().all():
                return False
            
            # 检查数值范围是否在哈希值范围内（0-999999）
            min_val = numeric_data.min()
            max_val = numeric_data.max()
            
            # 如果数值在哈希值范围内，且存在原始字符串映射，则认为是哈希值列
            if 0 <= min_val <= 999999 and 0 <= max_val <= 999999:
                # 检查session_state中是否有对应的原始字符串映射
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    return True
            
            return False
        except:
            return False
    
    def _get_original_strings(self, data: pd.DataFrame, col: str) -> list:
        """
        获取哈希值列对应的原始字符串
        
        Args:
            data: 数据框
            col: 列名
            
        Returns:
            list: 原始字符串列表
        """
        try:
            if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                return [hash_to_string.get(hash_val, str(hash_val)) for hash_val in data[col]]
            else:
                return data[col].astype(str).tolist()
        except:
            return data[col].astype(str).tolist()


class ZoomController:
    """
    缩放控制器类，负责图表的缩放功能
    """
    
    @staticmethod
    def initialize_zoom_state():
        """
        初始化缩放状态
        """
        if 'zoom_state' not in st.session_state:
            st.session_state.zoom_state = {
                'primary_auto': True,
                'secondary_auto': True,
                'third_auto': True,
                'primary_range': (0.0, 100.0),
                'secondary_range': (0.0, 100.0),
                'third_range': (0.0, 100.0)
            }
    
    @staticmethod
    def create_zoom_controls(axis_name: str, data_range: tuple) -> tuple:
        """
        创建缩放控制界面
        
        Args:
            axis_name: 轴名称
            data_range: 数据范围
            
        Returns:
            tuple: (是否自动缩放, 手动范围)
        """
        col1, col2 = st.columns([1, 3])
        
        with col1:
            auto_key = f"{axis_name}_auto"
            auto_scale = st.checkbox(
                f"{axis_name}轴自动缩放",
                value=st.session_state.zoom_state.get(auto_key, True),
                key=auto_key
            )
        
        with col2:
            if not auto_scale:
                range_key = f"{axis_name}_range"
                default_value = (float(data_range[0]), float(data_range[1]))
                # 确保从会话状态获取的值是元组格式
                stored_value = st.session_state.zoom_state.get(range_key, default_value)
                if isinstance(stored_value, list):
                    stored_value = (float(stored_value[0]), float(stored_value[1]))
                manual_range = st.slider(
                    f"{axis_name}轴范围",
                    min_value=float(data_range[0]),
                    max_value=float(data_range[1]),
                    value=stored_value,
                    step=(float(data_range[1]) - float(data_range[0])) / 100.0,
                    key=range_key
                )
                return auto_scale, manual_range
        
        return auto_scale, None
    
    @staticmethod
    def apply_zoom_to_figure(fig: go.Figure, zoom_configs: Dict[str, Any]):
        """
        将缩放配置应用到图表
        
        Args:
            fig: 图表对象
            zoom_configs: 缩放配置字典
        """
        # 应用主轴缩放
        if 'primary' in zoom_configs and not zoom_configs['primary']['auto']:
            fig.update_layout(
                yaxis=dict(range=zoom_configs['primary']['range'])
            )
        
        # 应用副轴缩放
        if 'secondary' in zoom_configs and not zoom_configs['secondary']['auto']:
            fig.update_layout(
                yaxis2=dict(range=zoom_configs['secondary']['range'])
            )
        
        # 应用第三轴缩放（如果存在）
        if 'third' in zoom_configs and not zoom_configs['third']['auto']:
            fig.update_layout(
                yaxis3=dict(range=zoom_configs['third']['range'])
            )