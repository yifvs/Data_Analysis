#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析工具库 - 提供各种数据分析功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class DataAnalysisTools:
    """
    数据分析工具类，提供各种统计分析功能
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def basic_statistics(self) -> Dict[str, Any]:
        """
        基础统计分析
        
        Returns:
            Dict: 包含基础统计信息的字典
        """
        result = {
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': {col: str(dtype) for col, dtype in self.data.dtypes.to_dict().items()},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # 数值型变量统计
        if self.numeric_columns:
            numeric_data = self.data[self.numeric_columns]
            
            def safe_to_dict(series):
                """安全地将pandas Series转换为可JSON序列化的字典"""
                result_dict = {}
                for key, value in series.to_dict().items():
                    if pd.isna(value):
                        result_dict[key] = None
                    elif np.isinf(value):
                        result_dict[key] = "infinity" if value > 0 else "-infinity"
                    else:
                        result_dict[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value
                return result_dict
            
            result['numeric_summary'] = {
                'count': safe_to_dict(numeric_data.count()),
                'mean': safe_to_dict(numeric_data.mean()),
                'std': safe_to_dict(numeric_data.std()),
                'min': safe_to_dict(numeric_data.min()),
                'max': safe_to_dict(numeric_data.max()),
                'median': safe_to_dict(numeric_data.median()),
                'q25': safe_to_dict(numeric_data.quantile(0.25)),
                'q75': safe_to_dict(numeric_data.quantile(0.75))
            }
        
        # 分类变量统计
        if self.categorical_columns:
            for col in self.categorical_columns:
                mode_series = self.data[col].mode()
                most_frequent = None
                if not mode_series.empty:
                    most_frequent_val = mode_series.iloc[0]
                    # 确保值可以JSON序列化
                    if pd.isna(most_frequent_val):
                        most_frequent = None
                    else:
                        most_frequent = str(most_frequent_val)
                
                # 处理value_counts的结果
                value_counts = self.data[col].value_counts().head(5)
                value_counts_dict = {}
                for key, value in value_counts.to_dict().items():
                    # 确保键和值都可以JSON序列化
                    safe_key = str(key) if not pd.isna(key) else "null"
                    safe_value = int(value) if isinstance(value, (np.integer, np.int64)) else value
                    value_counts_dict[safe_key] = safe_value
                
                result['categorical_summary'][col] = {
                    'unique_count': int(self.data[col].nunique()),
                    'most_frequent': most_frequent,
                    'value_counts': value_counts_dict
                }
        
        return result
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """
        相关性分析
        
        Returns:
            Dict: 相关性分析结果
        """
        if len(self.numeric_columns) < 2:
            return {'error': '需要至少2个数值型变量进行相关性分析'}
        
        numeric_data = self.data[self.numeric_columns]
        correlation_matrix = numeric_data.corr()
        
        # 找出强相关性（绝对值>0.7）
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'var1': str(correlation_matrix.columns[i]),
                        'var2': str(correlation_matrix.columns[j]),
                        'correlation': round(float(corr_value), 3)
                    })
        
        # 安全处理相关性矩阵
        corr_dict = {}
        for col in correlation_matrix.columns:
            corr_dict[col] = {}
            for row in correlation_matrix.index:
                value = correlation_matrix.loc[row, col]
                if pd.isna(value):
                    corr_dict[col][row] = None
                else:
                    corr_dict[col][row] = round(float(value), 3)
        
        # 安全处理平均相关性
        avg_corr = correlation_matrix.abs().mean().mean()
        safe_avg_corr = round(float(avg_corr), 3) if not pd.isna(avg_corr) else None
        
        return {
            'correlation_matrix': corr_dict,
            'strong_correlations': strong_correlations,
            'average_correlation': safe_avg_corr
        }
    
    def outlier_detection(self) -> Dict[str, Any]:
        """
        异常值检测
        
        Returns:
            Dict: 异常值检测结果
        """
        if not self.numeric_columns:
            return {'error': '没有数值型变量可进行异常值检测'}
        
        outliers = {}
        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) == 0:
                continue
                
            # IQR方法检测异常值
            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = data_col[(data_col < lower_bound) | (data_col > upper_bound)].index
            outlier_count = len(outlier_indices)
            outlier_percentage = (outlier_count / len(data_col)) * 100
            
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': round(float(outlier_percentage), 2),
                'lower_bound': round(float(lower_bound), 3) if not pd.isna(lower_bound) else None,
                'upper_bound': round(float(upper_bound), 3) if not pd.isna(upper_bound) else None
            }
        
        return outliers
    
    def trend_analysis(self) -> Dict[str, Any]:
        """
        趋势分析（针对时间序列或有序数据）
        
        Returns:
            Dict: 趋势分析结果
        """
        if not self.numeric_columns:
            return {'error': '没有数值型变量可进行趋势分析'}
        
        trends = {}
        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) < 3:
                continue
            
            # 计算趋势（使用线性回归斜率）
            x = np.arange(len(data_col))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data_col)
            
            # 判断趋势方向
            if slope > 0:
                trend_direction = '上升'
            elif slope < 0:
                trend_direction = '下降'
            else:
                trend_direction = '平稳'
            
            trends[col] = {
                'slope': round(float(slope), 6) if not pd.isna(slope) else None,
                'direction': trend_direction,
                'r_squared': round(float(r_value**2), 3) if not pd.isna(r_value) else None,
                'p_value': round(float(p_value), 6) if not pd.isna(p_value) else None,
                'significance': 'significant' if not pd.isna(p_value) and p_value < 0.05 else 'not_significant'
            }
        
        return trends
    
    def distribution_analysis(self) -> Dict[str, Any]:
        """
        分布分析
        
        Returns:
            Dict: 分布分析结果
        """
        if not self.numeric_columns:
            return {'error': '没有数值型变量可进行分布分析'}
        
        distributions = {}
        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) < 3:
                continue
            
            # 计算偏度和峰度
            skewness = stats.skew(data_col)
            kurtosis = stats.kurtosis(data_col)
            
            # 正态性检验
            if len(data_col) >= 8:  # Shapiro-Wilk test requires at least 3 samples
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(data_col[:5000])  # 限制样本数量
                    normality_test = {
                        'statistic': round(float(shapiro_stat), 4) if not pd.isna(shapiro_stat) else None,
                        'p_value': round(float(shapiro_p), 6) if not pd.isna(shapiro_p) else None,
                        'is_normal': bool(shapiro_p > 0.05) if not pd.isna(shapiro_p) else None
                    }
                except:
                    normality_test = {'error': '无法进行正态性检验'}
            else:
                normality_test = {'error': '样本量不足，无法进行正态性检验'}
            
            distributions[col] = {
                'skewness': round(float(skewness), 3) if not pd.isna(skewness) else None,
                'kurtosis': round(float(kurtosis), 3) if not pd.isna(kurtosis) else None,
                'normality_test': normality_test
            }
        
        return distributions
    
    def generate_insights(self) -> List[str]:
        """
        生成数据洞察
        
        Returns:
            List[str]: 洞察列表
        """
        insights = []
        
        # 基础统计洞察
        basic_stats = self.basic_statistics()
        insights.append(f"数据集包含 {basic_stats['shape'][0]} 行和 {basic_stats['shape'][1]} 列")
        
        # 缺失值洞察
        missing_total = sum(basic_stats['missing_values'].values())
        if missing_total > 0:
            insights.append(f"数据集存在 {missing_total} 个缺失值")
        
        # 相关性洞察
        if len(self.numeric_columns) >= 2:
            corr_analysis = self.correlation_analysis()
            if 'strong_correlations' in corr_analysis and corr_analysis['strong_correlations']:
                insights.append(f"发现 {len(corr_analysis['strong_correlations'])} 对强相关变量")
        
        # 异常值洞察
        if self.numeric_columns:
            outlier_analysis = self.outlier_detection()
            high_outlier_cols = [col for col, info in outlier_analysis.items() 
                               if isinstance(info, dict) and info.get('percentage', 0) > 5]
            if high_outlier_cols:
                insights.append(f"变量 {', '.join(high_outlier_cols)} 存在较多异常值")
        
        return insights


def analyze_data_with_tools(data: pd.DataFrame, analysis_type: str = 'comprehensive') -> Dict[str, Any]:
    """
    使用工具进行数据分析
    
    Args:
        data: 数据框
        analysis_type: 分析类型
        
    Returns:
        Dict: 分析结果
    """
    tools = DataAnalysisTools(data)
    
    if analysis_type == 'basic':
        return tools.basic_statistics()
    elif analysis_type == 'correlation':
        return tools.correlation_analysis()
    elif analysis_type == 'outliers':
        return tools.outlier_detection()
    elif analysis_type == 'trends':
        return tools.trend_analysis()
    elif analysis_type == 'distribution':
        return tools.distribution_analysis()
    elif analysis_type == 'comprehensive':
        return {
            'basic_statistics': tools.basic_statistics(),
            'correlation_analysis': tools.correlation_analysis(),
            'outlier_detection': tools.outlier_detection(),
            'trend_analysis': tools.trend_analysis(),
            'distribution_analysis': tools.distribution_analysis(),
            'insights': tools.generate_insights()
        }
    else:
        return {'error': f'未知的分析类型: {analysis_type}'}