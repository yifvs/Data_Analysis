import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re

# 设置页面布局为宽模式
st.set_page_config(layout="wide", page_title='振动值趋势', page_icon='🛫')

st.title(":red[发动机振动值趋势曲线]")

@st.cache_resource
def load_and_process_files(uploaded_files):
    list_left_fan_max = []
    list_left_lpt_max = []
    list_left_hpc_max = []
    list_left_hpt_max = []
    list_right_fan_max = []
    list_right_lpt_max = []
    list_right_hpc_max = []
    list_right_hpt_max = []
    list_date = []
    left_engine_table_data = []
    right_engine_table_data = []

    # 正则表达式匹配 YYYYMMDD_HHMMSS 格式
    pattern = re.compile(r"(\d{8}_\d{6})")

    def clean_data(df, columns):
        df = df[columns]
        for col in columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(axis=0, how='any', inplace=True)
        return df

    columns = ['DATE','ENG1N2','ENG2N2','ENG1VIBCN1','ENG1VIBTN1','ENG1VIBCN2','ENG1VIBTN2','ENG2VIBCN1','ENG2VIBTN1','ENG2VIBCN2','ENG2VIBTN2']

    for file in uploaded_files:
        file_name = file.name  # 提取文件名
        match = pattern.search(file_name)  # 匹配日期时间部分
        if match:
            long_string = match.group(1)
            try:
                time = datetime.strptime(long_string, "%Y%m%d_%H%M%S")
            except ValueError as e:
                st.error(f"Error parsing date from {long_string}: {e}")
                continue
        else:
            st.error(f"Filename {file_name} does not contain a valid date string.")
            continue

        # 读取CSV文件，跳过前4行作为header
        df = pd.read_csv(file, header=4, encoding='gb18030')
        df = clean_data(df, columns)  # 数据清洗
        if not df.empty:
            left_fan_max = df.loc[150:, 'ENG1VIBCN1'].max()
            left_lpt_max = df.loc[150:, 'ENG1VIBTN1'].max()
            left_hpc_max = df.loc[150:, 'ENG1VIBCN2'].max()
            left_hpt_max = df.loc[150:, 'ENG1VIBTN2'].max()
            right_fan_max = df.loc[150:, 'ENG2VIBCN1'].max()
            right_lpt_max = df.loc[150:, 'ENG2VIBTN1'].max()
            right_hpc_max = df.loc[150:, 'ENG2VIBCN2'].max()
            right_hpt_max = df.loc[150:, 'ENG2VIBTN2'].max()

            list_left_fan_max.append(left_fan_max)
            list_left_lpt_max.append(left_lpt_max)
            list_left_hpc_max.append(left_hpc_max)
            list_left_hpt_max.append(left_hpt_max)
            list_right_fan_max.append(right_fan_max)
            list_right_lpt_max.append(right_lpt_max)
            list_right_hpc_max.append(right_hpc_max)
            list_right_hpt_max.append(right_hpt_max)
            list_date.append(time)

            left_engine_table_data.append({
                'Time': time,
                'FAN': left_fan_max,
                'LPT': left_lpt_max,
                'HPC': left_hpc_max,
                'HPT': left_hpt_max
            })

            right_engine_table_data.append({
                'Time': time,
                'FAN': right_fan_max,
                'LPT': right_lpt_max,
                'HPC': right_hpc_max,
                'HPT': right_hpt_max
            })

    if list_date:
        df = pd.DataFrame({
            'Left_FAN_MAX': list_left_fan_max,
            'Left_LPT_MAX': list_left_lpt_max,
            'Left_HPC_MAX': list_left_hpc_max,
            'Left_HPT_MAX': list_left_hpt_max,
            'Right_FAN_MAX': list_right_fan_max,
            'Right_LPT_MAX': list_right_lpt_max,
            'Right_HPC_MAX': list_right_hpc_max,
            'Right_HPT_MAX': list_right_hpt_max,
            'time': list_date
        })
        df.sort_values(by=['time'], ascending=True, inplace=True)
        return df, left_engine_table_data, right_engine_table_data
    else:
        return None, [], []

# 上传文件组件
uploaded_files = st.file_uploader(":blue[📁 请同时选中并拖拽多个CSV文档批量上传自动译码文件]", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    df, left_engine_table_data, right_engine_table_data = load_and_process_files(uploaded_files)

    if df is not None:
        # 使用st.columns创建两列布局
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("左发振动值")
            left_engine_df = pd.DataFrame(left_engine_table_data)
            st.dataframe(left_engine_df, use_container_width=True)

        with col2:
            # 左发振动值曲线
            fig_left_engine = px.line(
                df, x='time', y=[
                    'Left_FAN_MAX', 'Left_LPT_MAX', 'Left_HPC_MAX', 'Left_HPT_MAX'
                ],
                title='左发振动值变化趋势',
                labels={'value': 'Vibration', 'variable': 'Sensor'},
                hover_data={'time': '|%B %d, %Y %I:%M %p'}
            )
            st.plotly_chart(fig_left_engine, use_container_width=True)

        col3, col4 = st.columns([1, 1])

        with col3:
            st.subheader("右发振动值")
            right_engine_df = pd.DataFrame(right_engine_table_data)
            st.dataframe(right_engine_df, use_container_width=True)

        with col4:
            # 右发振动值曲线
            fig_right_engine = px.line(
                df, x='time', y=[
                    'Right_FAN_MAX', 'Right_LPT_MAX', 'Right_HPC_MAX', 'Right_HPT_MAX'
                ],
                title='右发振动值变化趋势',
                labels={'value': 'Vibration', 'variable': 'Sensor'},
                hover_data={'time': '|%B %d, %Y %I:%M %p'}
            )
            st.plotly_chart(fig_right_engine, use_container_width=True)
    else:
        st.warning("No valid data found in the uploaded files.")
else:
    st.info("当前仅支持737NG机型的可视化，请上传自动译码的csv文件。")




