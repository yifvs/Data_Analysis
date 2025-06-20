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
    
    def create_custom_axis_chart(self, data: pd.DataFrame, custom_config: Dict[str, Any]) -> go.Figure:
        """
        根据自定义轴配置创建图表
        
        Args:
            data: 数据框
            custom_config: 自定义轴配置
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        x_column = custom_config.get('x_column')
        y_columns = custom_config.get('y_columns', [])
        chart_type = custom_config.get('chart_type', 'line')
        color_theme = custom_config.get('color_theme', 'plotly')
        
        if not x_column or x_column is None or not y_columns:
            # 创建空图表
            fig = go.Figure()
            fig.add_annotation(
                text="请选择X轴和Y轴数据列",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # 创建图表
        fig = go.Figure()
        
        # 获取X轴数据
        x_data = data[x_column]
        
        # 设置颜色序列
        if color_theme == 'plotly':
            colors = px.colors.qualitative.Plotly
        else:
            colors = getattr(px.colors.sequential, color_theme.title(), px.colors.qualitative.Plotly)
        
        # 为每个Y轴列创建trace
        for i, y_col in enumerate(y_columns):
            y_data = data[y_col]
            color = colors[i % len(colors)]
            
            # 检查是否为哈希值列
            is_hash_column = self._is_hash_column(data, y_col)
            
            if chart_type == 'line':
                if is_hash_column:
                    # 获取原始字符串数据用于悬停显示
                    original_strings = self._get_original_strings(data, y_col)
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines+markers',
                        name=y_col,
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        customdata=original_strings,
                        hovertemplate=f'<b>{y_col}</b><br>' +
                                    f'{custom_config.get("x_title", x_column)}: %{{x}}<br>' +
                                    f'原始值: %{{customdata}}<br>' +
                                    f'哈希值: %{{y}}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines+markers',
                        name=y_col,
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        hovertemplate=f'<b>{y_col}</b><br>' +
                                    f'{custom_config.get("x_title", x_column)}: %{{x}}<br>' +
                                    f'{custom_config.get("y_title", "数值")}: %{{y}}<extra></extra>'
                    ))
            
            elif chart_type == 'bar':
                fig.add_trace(go.Bar(
                    x=x_data,
                    y=y_data,
                    name=y_col,
                    marker_color=color,
                    hovertemplate=f'<b>{y_col}</b><br>' +
                                f'{custom_config.get("x_title", x_column)}: %{{x}}<br>' +
                                f'{custom_config.get("y_title", "数值")}: %{{y}}<extra></extra>'
                ))
            
            elif chart_type == 'scatter':
                if is_hash_column:
                    original_strings = self._get_original_strings(data, y_col)
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name=y_col,
                        marker=dict(color=color, size=8),
                        customdata=original_strings,
                        hovertemplate=f'<b>{y_col}</b><br>' +
                                    f'{custom_config.get("x_title", x_column)}: %{{x}}<br>' +
                                    f'原始值: %{{customdata}}<br>' +
                                    f'哈希值: %{{y}}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name=y_col,
                        marker=dict(color=color, size=8),
                        hovertemplate=f'<b>{y_col}</b><br>' +
                                    f'{custom_config.get("x_title", x_column)}: %{{x}}<br>' +
                                    f'{custom_config.get("y_title", "数值")}: %{{y}}<extra></extra>'
                    ))
            
            elif chart_type == 'area':
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=y_col,
                    fill='tonexty' if i > 0 else 'tozeroy',
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>{y_col}</b><br>' +
                                f'{custom_config.get("x_title", x_column)}: %{{x}}<br>' +
                                f'{custom_config.get("y_title", "数值")}: %{{y}}<extra></extra>'
                ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{custom_config.get('y_title', '数值')} vs {custom_config.get('x_title', x_column)}",
                x=0.5,
                font=dict(size=16, color='#2E86AB')
            ),
            xaxis=dict(
                title=custom_config.get('x_title', x_column),
                showgrid=custom_config.get('show_grid', True),
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                title=custom_config.get('y_title', '数值'),
                showgrid=custom_config.get('show_grid', True),
                gridcolor='lightgray',
                gridwidth=1
            ),
            hovermode='x unified',
            template='plotly_white',
            showlegend=len(y_columns) > 1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 应用轴范围设置
        if not custom_config.get('x_range_auto', True) and custom_config.get('x_range'):
            fig.update_xaxes(range=custom_config['x_range'])
        
        if not custom_config.get('y_range_auto', True) and custom_config.get('y_range'):
            fig.update_yaxes(range=custom_config['y_range'])
        
        return fig
    
    def create_single_axis_chart(self, data: pd.DataFrame, selected_columns: List[str], 
                                 chart_type: str = 'line', animation_frames: int = None) -> go.Figure:
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
            elif chart_type == 'animated':
                # 创建动态图表（时间序列动画）
                return self._create_animated_chart(data, selected_columns, animation_frames)
        
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
            # 处理NA值：使用前向填充，如果仍有NA则用0填充
            y_values = y_values.ffill().fillna(0)
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
            # 处理NA值：使用前向填充，如果仍有NA则用0填充
            y_values = y_values.ffill().fillna(0)
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
        
        # 自动设置Y轴范围以匹配数据范围
        if data is not None and columns:
            # 计算所有选中列的数值范围
            numeric_data = []
            for col in columns:
                if col in data.columns:
                    col_data = pd.to_numeric(data[col], errors='coerce').dropna()
                    if not col_data.empty:
                        numeric_data.extend(col_data.tolist())
            
            if numeric_data:
                y_min = min(numeric_data)
                y_max = max(numeric_data)
                y_range = y_max - y_min
                # 添加10%的边距以确保数据点不贴边
                margin = y_range * 0.1 if y_range > 0 else 1
                yaxis_config.update({
                    'range': [y_min - margin, y_max + margin],
                    'autorange': False
                })
            
            # 为哈希值列设置自定义Y轴刻度
            hash_cols = [col for col in columns if self._is_hash_column(data, col)]
            if hash_cols:
                col = hash_cols[0]  # 使用第一个哈希值列
                unique_hashes = sorted(data[col].dropna().unique())
                if hasattr(st.session_state, 'string_mappings') and col in st.session_state.string_mappings:
                    hash_to_string = {v: k for k, v in st.session_state.string_mappings[col].items()}
                    yaxis_config.update({
                        'tickmode': 'array',
                        'tickvals': unique_hashes,
                        'ticktext': [hash_to_string.get(h, str(h)) for h in unique_hashes],
                        'autorange': False
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
    
    def _create_animated_chart(self, data: pd.DataFrame, selected_columns: List[str], 
                              animation_frames: int = None) -> go.Figure:
        """
        创建优化的动态时间序列图表
        
        Args:
            data: 数据框
            selected_columns: 选中的列
            animation_frames: 动画帧数
            
        Returns:
            plotly.graph_objects.Figure: 动态图表对象
        """
        frames = []
        steps = []
        
        # 使用用户选择的帧数，如果为None则使用所有数据点
        if animation_frames is None:
            # 使用所有数据点
            frame_indices = list(range(len(data)))
        else:
            # 使用指定的帧数
            max_frames = animation_frames
            if len(data) > max_frames:
                step = len(data) // max_frames
                frame_indices = list(range(0, len(data), step))
                if frame_indices[-1] != len(data) - 1:
                    frame_indices.append(len(data) - 1)
            else:
                frame_indices = list(range(len(data)))
        
        # 确保frame_indices不为空且至少有一个有效索引
        if not frame_indices:
            frame_indices = [0]
        
        # 确保最后一帧包含所有数据
        if frame_indices[-1] != len(data) - 1:
            frame_indices.append(len(data) - 1)
        
        # 为每一帧创建数据
        for frame_idx, i in enumerate(frame_indices):
            # 确保第一帧至少包含2个数据点以形成线条
            if frame_idx == 0 and i == 0 and len(data) > 1:
                frame_data = data.iloc[:2]  # 第一帧至少显示前两个点
            else:
                frame_data = data.iloc[:i+1]  # 包含当前索引的数据
            frame_traces = []
            
            # 为每个列创建trace
            for j, col in enumerate(selected_columns):
                color = self.colors[j % len(self.colors)]
                is_hash_column = self._is_hash_column(data, col)
                
                # 获取数值数据并处理NA值
                y_values = pd.to_numeric(frame_data[col], errors='coerce')
                # 处理NA值：使用前向填充，如果仍有NA则用0填充
                y_values = y_values.ffill().fillna(0)
                
                # 根据数据点数量决定显示模式 - 用户要求只显示线条
                display_mode = 'lines' if len(frame_data) >= 2 else 'markers'
                
                if is_hash_column:
                    original_strings = self._get_original_strings(frame_data, col)
                    frame_traces.append(go.Scatter(
                        x=frame_data.index,
                        y=y_values,
                        mode=display_mode,
                        name=col,
                        line=dict(color=color, width=1.5, shape='spline') if len(frame_data) >= 2 else None,
                        marker=dict(
                            color=color, 
                            size=8 if len(frame_data) < 2 else 6,
                            line=dict(width=1, color='white'),
                            symbol='circle'
                        ),
                        customdata=original_strings,
                        hovertemplate=f'<b>{col}</b><br>' +
                                    '索引: %{x}<br>' +
                                    '原始值: %{customdata}<br>' +
                                    '数值: %{y:.2f}<extra></extra>',
                        legendgroup=col,  # 图例分组
                        showlegend=(frame_idx == 0)  # 只在第一帧显示图例
                    ))
                    
                    # 添加动态悬停标记点（跟随曲线的最新数据点）
                    if len(frame_data) > 0:
                        latest_x = frame_data.index[-1]
                        latest_y = y_values.iloc[-1]
                        latest_original = original_strings[-1] if original_strings else str(latest_y)
                        
                        frame_traces.append(go.Scatter(
                            x=[latest_x],
                            y=[latest_y],
                            mode='markers',
                            name=f'{col}_highlight',
                            marker=dict(
                                color=color,
                                size=15,
                                line=dict(width=3, color='white'),
                                symbol='circle',
                                opacity=0.8
                            ),
                            customdata=[latest_original],
                            hovertemplate=f'<b>🎯 {col} 当前值</b><br>' +
                                        '索引: %{x}<br>' +
                                        '原始值: %{customdata}<br>' +
                                        '数值: %{y:.2f}<extra></extra>',
                            showlegend=False,
                            hoverinfo='all'
                        ))
                        
                        # 添加始终可见的文本标注（显示当前数据点信息）
                        text_display = f"{col}: {latest_y:.2f}"
                        if is_hash_column and latest_original != str(latest_y):
                            text_display = f"{col}: {latest_original} ({latest_y:.2f})"
                        
                        frame_traces.append(go.Scatter(
                            x=[latest_x],
                            y=[latest_y],
                            mode='markers+text',
                            name=f'{col}_text',
                            marker=dict(
                                color='rgba(0,0,0,0)',  # 透明标记
                                size=1
                            ),
                            text=[text_display],
                            textposition='top center',
                            textfont=dict(
                                size=12,
                                color=color,
                                family='Arial Black'
                            ),
                            showlegend=False,
                            hoverinfo='skip'  # 跳过悬停信息，避免重复
                        ))
                else:
                    frame_traces.append(go.Scatter(
                        x=frame_data.index,
                        y=y_values,
                        mode=display_mode,
                        name=col,
                        line=dict(color=color, width=1.5, shape='spline') if len(frame_data) >= 2 else None,
                        marker=dict(
                            color=color, 
                            size=8 if len(frame_data) < 2 else 6,
                            line=dict(width=1, color='white'),
                            symbol='circle'
                        ),
                        hovertemplate=f'<b>{col}</b><br>' +
                                    '索引: %{x}<br>' +
                                    '数值: %{y:.2f}<extra></extra>',
                        legendgroup=col,  # 图例分组
                        showlegend=(frame_idx == 0)  # 只在第一帧显示图例
                    ))
                    
                    # 添加动态悬停标记点（跟随曲线的最新数据点）
                    if len(frame_data) > 0:
                        latest_x = frame_data.index[-1]
                        latest_y = y_values.iloc[-1]
                        
                        frame_traces.append(go.Scatter(
                            x=[latest_x],
                            y=[latest_y],
                            mode='markers',
                            name=f'{col}_highlight',
                            marker=dict(
                                color=color,
                                size=15,
                                line=dict(width=3, color='white'),
                                symbol='circle',
                                opacity=0.8
                            ),
                            hovertemplate=f'<b>🎯 {col} 当前值</b><br>' +
                                        '索引: %{x}<br>' +
                                        '数值: %{y:.2f}<extra></extra>',
                            showlegend=False,
                            hoverinfo='all'
                        ))
                        
                        # 添加始终可见的文本标注（显示当前数据点信息）
                        text_display = f"{col}: {latest_y:.2f}"
                        
                        frame_traces.append(go.Scatter(
                            x=[latest_x],
                            y=[latest_y],
                            mode='markers+text',
                            name=f'{col}_text',
                            marker=dict(
                                color='rgba(0,0,0,0)',  # 透明标记
                                size=1
                            ),
                            text=[text_display],
                            textposition='top center',
                            textfont=dict(
                                size=12,
                                color=color,
                                family='Arial Black'
                            ),
                            showlegend=False,
                            hoverinfo='skip'  # 跳过悬停信息，避免重复
                        ))
            
            frames.append(go.Frame(
                data=frame_traces,
                name=str(frame_idx)
            ))
            
            # 创建更简洁的步骤标签（移除%符号避免重复）
            progress_percent = int((frame_idx / (len(frame_indices) - 1)) * 100)
            steps.append(dict(
                args=[[str(frame_idx)], {"frame": {"duration": 500, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 150}}],
                label=str(progress_percent),
                method="animate"
            ))
        
        # 创建初始图表 - 确保显示第一帧的数据
        initial_data = []
        if frames:
            # 使用第一帧的数据作为初始显示
            initial_data = frames[0].data
        
        fig = go.Figure(
            data=initial_data,
            frames=frames
        )
        
        # 添加优化的播放控制按钮
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 600, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 300,
                                        "easing": "quadratic-in-out"}}],
                        "label": "🎬 播放",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "⏸️ 暂停",
                        "method": "animate"
                    },
                    {
                        "args": [[frames[0].name], {"frame": {"duration": 0, "redraw": True},
                                                   "mode": "immediate",
                                                   "transition": {"duration": 0}}],
                        "label": "⏮️ 重置",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.02,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "rgba(0,0,0,0.2)",
                "borderwidth": 1
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14, "color": "#2E86AB"},
                    "prefix": "进度: ",
                    "suffix": "%",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 200, "easing": "cubic-in-out"},
                "pad": {"b": 20, "t": 20},
                "len": 0.85,
                "x": 0.1,
                "y": 0,
                "steps": steps,
                "bgcolor": "rgba(46, 134, 171, 0.1)",
                "bordercolor": "rgba(46, 134, 171, 0.3)",
                "borderwidth": 1,
                "tickcolor": "rgba(46, 134, 171, 0.6)"
            }]
        )
        
        # 为动态图表设置特殊的布局配置（不使用通用的_apply_layout_config）
        # 计算Y轴范围
        yaxis_config = dict(**self.chart_config['grid_config'])
        
        # 自动设置Y轴范围以匹配数据范围
        if data is not None and selected_columns:
            # 计算所有选中列的数值范围
            numeric_data = []
            for col in selected_columns:
                if col in data.columns:
                    col_data = pd.to_numeric(data[col], errors='coerce').dropna()
                    if not col_data.empty:
                        numeric_data.extend(col_data.tolist())
            
            if numeric_data:
                y_min = min(numeric_data)
                y_max = max(numeric_data)
                y_range = y_max - y_min
                # 添加10%的边距以确保数据点不贴边
                margin = y_range * 0.1 if y_range > 0 else 1
                yaxis_config.update({
                    'range': [y_min - margin, y_max + margin],
                    'autorange': False
                })
        
        # 更新布局以适应动画和优化视觉效果
        fig.update_layout(
            title={
                'text': "🎬 动态时间序列图表",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E86AB'}
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            width=self.chart_config['default_width'],
            height=self.chart_config['default_height'] + 120,
            margin=dict(t=80, b=80, l=60, r=60),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#333333"),
            hovermode='x',
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                font_size=12,
                align="left"
            ),
            # 为动态图表设置合适的x轴配置
            xaxis=dict(
                **self.chart_config['grid_config'],
                # 自动设置x轴范围以适应数据索引
                range=[0, len(data) - 1],
                autorange=False,
                # 根据数据长度动态设置刻度间隔
                tickmode='linear',
                dtick=max(1, len(data) // 10),  # 大约显示10个刻度
                tickangle=45
            ),
            yaxis=yaxis_config
        )
        
        # 优化网格和轴样式
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.3)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.3)'
        )
        
        return fig


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