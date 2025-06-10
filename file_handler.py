# -*- coding: utf-8 -*-
"""
文件处理模块 - 负责文件读取、编码检测和数据预处理
"""

import pandas as pd
import streamlit as st
from typing import Optional, Tuple
from config import FILE_CONFIG, TIME_COLUMN_NAMES


class FileProcessor:
    """文件处理器类，负责文件读取和数据预处理"""
    
    def __init__(self):
        self.encoding_options = FILE_CONFIG['encoding_options']
        self.time_columns = TIME_COLUMN_NAMES
    
    def detect_encoding_and_read(self, file, file_ext: str, header_row: int) -> Optional[pd.DataFrame]:
        """
        检测文件编码并读取数据
        
        Args:
            file: 上传的文件对象
            file_ext: 文件扩展名
            header_row: 表头行数
            
        Returns:
            pandas.DataFrame: 读取的数据框，失败时返回None
        """
        file.seek(0)
        
        if file_ext == "csv":
            return self._read_csv_with_encoding(file, header_row)
        elif file_ext == "xlsx":
            return self._read_excel(file, header_row)
        else:
            st.error(f"不支持的文件格式：{file_ext}")
            return None
    
    def _read_csv_with_encoding(self, file, header_row: int) -> Optional[pd.DataFrame]:
        """
        使用多种编码尝试读取CSV文件
        
        Args:
            file: CSV文件对象
            header_row: 表头行数
            
        Returns:
            pandas.DataFrame: 读取的数据框，失败时返回None
        """
        for encoding in self.encoding_options:
            try:
                file.seek(0)
                data = pd.read_csv(file, header=int(header_row), dtype='str', encoding=encoding)
                if not data.empty and len(data.columns) > 0:
                    st.info(f"✅ 使用编码：{encoding}")
                    return data
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
        
        st.error("CSV文件读取失败：无法使用任何编码读取文件")
        return None
    
    def _read_excel(self, file, header_row: int) -> Optional[pd.DataFrame]:
        """
        读取Excel文件
        
        Args:
            file: Excel文件对象
            header_row: 表头行数
            
        Returns:
            pandas.DataFrame: 读取的数据框，失败时返回None
        """
        try:
            file.seek(0)
            data = pd.read_excel(file, header=int(header_row), dtype='str')
            if data.empty or len(data.columns) == 0:
                raise ValueError("Excel文件为空或无有效列")
            return data
        except Exception as e:
            st.error(f"Excel文件读取失败：{str(e)}")
            return None
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据的有效性
        
        Args:
            data: 要验证的数据框
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            st.error("❌ 文件为空或没有有效数据")
            return False
        
        if len(data.columns) == 0:
            st.error("❌ 文件中没有检测到列，请检查表头行设置")
            return False
        
        st.success(f"📊 成功读取文件，共 {len(data)} 行，{len(data.columns)} 列")
        return True
    
    def detect_time_column(self, data: pd.DataFrame) -> Optional[str]:
        """
        自动检测时间列
        
        Args:
            data: 数据框
            
        Returns:
            str: 检测到的时间列名，未找到时返回None
        """
        for time_col in self.time_columns:
            if time_col in data.columns:
                return time_col
        return None
    
    def set_index_column(self, data: pd.DataFrame, index_col: str) -> pd.DataFrame:
        """
        设置索引列
        
        Args:
            data: 数据框
            index_col: 索引列名
            
        Returns:
            pandas.DataFrame: 设置索引后的数据框
        """
        try:
            return data.set_index(index_col)
        except Exception as e:
            st.error(f"设置索引列失败：{str(e)}")
            return data
    
    def process_file_with_options(self, file, file_ext: str, header_row: int) -> Optional[pd.DataFrame]:
        """
        完整的文件处理流程，包括索引列处理
        
        Args:
            file: 上传的文件对象
            file_ext: 文件扩展名
            header_row: 表头行数
            
        Returns:
            pandas.DataFrame: 处理后的数据框
        """
        # 读取数据
        data = self.detect_encoding_and_read(file, file_ext, header_row)
        
        if not self.validate_data(data):
            return None
        
        # 检测时间列
        time_column = self.detect_time_column(data)
        
        if time_column:
            st.info(f"✅ 自动检测到时间列：{time_column}，将其设为索引列")
            return self.set_index_column(data, time_column)
        else:
            # 提供索引选择选项
            return self._handle_index_selection(data)
    
    def _handle_index_selection(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        处理用户的索引列选择
        
        Args:
            data: 数据框
            
        Returns:
            pandas.DataFrame: 处理后的数据框
        """
        st.warning("⚠️ 未检测到标准时间列，请选择索引处理方式：")
        
        index_option = st.radio(
            "索引列选择：",
            ["使用默认数字索引", "手动选择列作为索引", "使用第一列作为索引"],
            key="index_selection"
        )
        
        if index_option == "使用默认数字索引":
            st.success("✅ 使用默认数字索引")
            return data
        
        elif index_option == "手动选择列作为索引":
            selected_index_col = st.selectbox(
                "请选择要用作索引的列：",
                data.columns.tolist(),
                key="manual_index_selection"
            )
            
            if st.button("确认使用选定的索引列", key="confirm_index"):
                st.success(f"✅ 使用 {selected_index_col} 作为索引列")
                return self.set_index_column(data, selected_index_col)
            else:
                st.info("👆 请点击确认按钮以应用索引列设置")
                return None
        
        elif index_option == "使用第一列作为索引":
            first_col = data.columns[0]
            st.success(f"✅ 使用第一列 '{first_col}' 作为索引")
            return self.set_index_column(data, first_col)
        
        return None


def clean_data(data: pd.DataFrame, skip_before: int = 0, skip_after: int = 0) -> pd.DataFrame:
    """
    清理数据，删除指定的前后行数
    
    Args:
        data: 原始数据框
        skip_before: 前部删除行数
        skip_after: 尾部删除行数
        
    Returns:
        pandas.DataFrame: 清理后的数据框
    """
    if skip_before > 0:
        data = data.iloc[skip_before:]
    
    if skip_after > 0:
        data = data.iloc[:-skip_after]
    
    return data.reset_index(drop=True)