# -*- coding: utf-8 -*-
"""
配置模块 - 存储应用程序的配置常量和设置
"""

import plotly.express as px

# 应用程序基本配置
APP_CONFIG = {
    'page_title': 'Data Analysis v2.2.0',
    'page_icon': '📊',
    'layout': 'wide'
}

# 颜色配置
COLOR_SCHEMES = {
    'default': px.colors.qualitative.Set1,
    'plotly': px.colors.qualitative.Plotly,
    'pastel': px.colors.qualitative.Pastel1,
    'dark': px.colors.qualitative.Dark2
}

# 文件上传配置
FILE_CONFIG = {
    'supported_types': ['csv', 'xlsx'],
    'max_file_size': 200,  # MB
    'encoding_options': ['gb18030', 'utf-8', 'gbk', 'utf-8-sig']
}

# 时间列检测配置
TIME_COLUMN_NAMES = [
    'Time', 'DateTime', 'Timestamp', '时间',
    'time', 'datetime', 'timestamp', 'date',
    'Date', 'TIME', 'DATETIME', 'TIMESTAMP'
]

# DeepSeek API 配置
DEEPSEEK_CONFIG = {
    'api_url': 'https://api.deepseek.com/v1/chat/completions',
    'default_model': 'deepseek-chat',
    'temperature': 0.7,
    'max_tokens': 4000
}

# LLM 系统提示模板
SYSTEM_TEMPLATE = """
你是一个专业的数据分析专家。你的任务是：
- 理解用户关于数据的问题
- 执行适当的数据分析
- 提供清晰的数据解释和洞察
- 回答数据相关的统计问题

可用的数据列：{columns}
数据概览：
- 行数：{rows}
- 列数：{cols}
- 数据样本：
{sample}

请用中文回答，保持专业、简洁和有洞察力。专注于数据分析。
"""

# 图表配置
CHART_CONFIG = {
    'default_width': 1200,
    'default_height': 600,
    'grid_config': {
        'showgrid': True,
        'gridwidth': 1,
        'gridcolor': 'lightgray',
        'showline': True,
        'linewidth': 1,
        'linecolor': 'black'
    },
    'rangeslider_config': {
        'visible': True,
        'thickness': 0.1
    }
}

# 版权信息
COPYRIGHT_INFO = {
    'text': 'Copyright © 2023-2026 海航航空技术有限公司. All Rights Reserved.',
    'description': '本应用程序受著作权法和其他知识产权法保护。未经授权，禁止复制、修改或分发本程序的任何部分。',
    'email': 'kangy_wang@hnair.com'
}

BUG_REPORT_EMAIL = "kangy_wang@hnair.com"
