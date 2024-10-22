import streamlit as st
import pandas as pd
from datetime import datetime
import re
import plotly.graph_objects as go

# 设置页面布局
st.set_page_config(layout='wide', page_title='起落架收放时间趋势', page_icon='🛫')

# 处理737NG起落架收放时间的函数
def process_737ng_file(uploaded_file):
    pattern = re.compile(r"(\d{8}_\d{6})")
    file_name = uploaded_file.name
    match = pattern.search(file_name)
    if match:
        long_string = match.group(1)
        try:
            time = datetime.strptime(long_string, "%Y%m%d_%H%M%S")
        except ValueError as e:
            print(f"Error parsing date from {long_string}: {e}")
            return None
    else:
        print(f"Filename {file_name} does not contain a valid date string.")
        return None

    df = pd.read_csv(uploaded_file, header=4, encoding='gb18030')
    df['ENG1N2'] = pd.to_numeric(df['ENG1N2'], errors='coerce')
    df = df[(df.ENG1N2 > 60) & (df.ENG1N2 < 100)]

    # 筛选爬升阶段的数据
    df1 = df[(df.FLIGHT_PHASE == 'INIT CLIMB') | (df.FLIGHT_PHASE == 'CRUISE') | (df.FLIGHT_PHASE == 'LVL CHANGE')]
    left_gear_bool = (df1['GEAR_WAR1'] == 'WARNING')
    right_gear_bool = (df1['GEAR_WAR2'] == 'WARNING')
    nose_gear_bool = (df1['GEAR_NOS_WAR'] == 'WARNING')

    # 筛选下降阶段的数据
    df2 = df[(df.FLIGHT_PHASE == 'APPROACH') | (df.FLIGHT_PHASE == 'FINAL APP') | (df.FLIGHT_PHASE == 'DESCENT')]
    left_gear_bool2 = (df2['GEAR_WAR1'] == 'WARNING')
    right_gear_bool2 = (df2['GEAR_WAR2'] == 'WARNING')
    nose_gear_bool2 = (df2['GEAR_NOS_WAR'] == 'WARNING')

    summary = pd.DataFrame({
        'time': [time],
        '左主起收上': [left_gear_bool.sum()],
        '右主起收上': [right_gear_bool.sum()],
        '前起收上': [nose_gear_bool.sum()],
        '左主起放下': [left_gear_bool2.sum()],
        '右主起放下': [right_gear_bool2.sum()],
        '前起放下': [nose_gear_bool2.sum()]
    })

    return summary

# 处理737MAX起落架收放时间的函数
def process_737max_file(uploaded_file):
    pattern = re.compile(r"(\d{8}_\d{6})")
    file_name = uploaded_file.name
    match = pattern.search(file_name)
    if match:
        long_string = match.group(1)
        try:
            time = datetime.strptime(long_string, "%Y%m%d_%H%M%S")
        except ValueError as e:
            print(f"Error parsing date from {long_string}: {e}")
            return None
    else:
        print(f"Filename {file_name} does not contain a valid date string.")
        return None

    df = pd.read_csv(uploaded_file, header=4, encoding='gb18030')
    df['N12_C'] = pd.to_numeric(df['N12_C'], errors='coerce')
    df = df[(df.N12_C > 20) & (df.N12_C < 100)]

    # 筛选爬升阶段的数据
    df1 = df[(df.FLIGHT_PHASE == 'INIT CLIMB') | (df.FLIGHT_PHASE == 'CRUISE') | (df.FLIGHT_PHASE == 'LVL CHANGE')]
    left_gear_bool = (df1['LEFT_RED_WARN'] == 'LIGHT ON')
    right_gear_bool = (df1['RIGHT_RED_WARN'] == 'LIGHT ON')
    nose_gear_bool = (df1['NOSE_RED_WARN'] == 'LIGHT ON')

    # 筛选下降阶段的数据
    df2 = df[(df.FLIGHT_PHASE == 'APPROACH') | (df.FLIGHT_PHASE == 'FINAL APP') | (df.FLIGHT_PHASE == 'DESCENT')]
    left_gear_bool2 = (df2['LEFT_RED_WARN'] == 'LIGHT ON')
    right_gear_bool2 = (df2['RIGHT_RED_WARN'] == 'LIGHT ON')
    nose_gear_bool2 = (df2['NOSE_RED_WARN'] == 'LIGHT ON')

    summary = pd.DataFrame({
        'time': [time],
        '左主起收上': [left_gear_bool.sum()],
        '右主起收上': [right_gear_bool.sum()],
        '前起收上': [nose_gear_bool.sum()],
        '左主起放下': [left_gear_bool2.sum()],
        '右主起放下': [right_gear_bool2.sum()],
        '前起放下': [nose_gear_bool2.sum()]
    })

    return summary

# 主函数，用于设置 Streamlit 应用
def main():
    st.title(':blue[起落架收放时间趋势分析]')

    # 添加机型选择下拉框
    aircraft_model = st.selectbox("请选择机型", ("737NG", "737MAX"))

    uploaded_files = st.file_uploader("📁 请同时选中并拖拽多个CSV文档批量上传自动译码文件", type=["csv"], accept_multiple_files=True)

    if uploaded_files is not None:
        all_data = pd.DataFrame()

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            # 根据选择的机型调用不同的处理函数
            if aircraft_model == "737NG":
                file_data = process_737ng_file(uploaded_file)
            else:
                file_data = process_737max_file(uploaded_file)

            # 在合并 file_data 到 all_data 之前，检查 file_data 是否为 None，并确保它包含 time 列
            if file_data is not None and 'time' in file_data.columns:
                all_data = pd.concat([all_data, file_data], ignore_index=True)

        all_data.sort_values(by='time', inplace=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('数据表格')
            st.write(all_data)

        with col2:
            st.subheader('趋势图')

            if not all_data.empty:
                # 使用 plotly.graph_objects 来创建带有不同形状的折线图
                fig = go.Figure()

                # 为每个数据列分别添加折线和不同的形状
                fig.add_trace(go.Scatter(x=all_data['time'], y=all_data['左主起收上'],
                                         mode='markers+lines',
                                         name='左主起收上',
                                         marker=dict(symbol='circle', size=10)))  # 圆形点

                fig.add_trace(go.Scatter(x=all_data['time'], y=all_data['右主起收上'],
                                         mode='markers+lines',
                                         name='右主起收上',
                                         marker=dict(symbol='square', size=10)))  # 方形点

                fig.add_trace(go.Scatter(x=all_data['time'], y=all_data['前起收上'],
                                         mode='markers+lines',
                                         name='前起收上',
                                         marker=dict(symbol='triangle-up', size=10)))  # 三角形点

                fig.add_trace(go.Scatter(x=all_data['time'], y=all_data['左主起放下'],
                                         mode='markers+lines',
                                         name='左主起放下',
                                         marker=dict(symbol='diamond', size=10)))  # 菱形点

                fig.add_trace(go.Scatter(x=all_data['time'], y=all_data['右主起放下'],
                                         mode='markers+lines',
                                         name='右主起放下',
                                         marker=dict(symbol='cross', size=10)))  # 十字形点

                fig.add_trace(go.Scatter(x=all_data['time'], y=all_data['前起放下'],
                                         mode='markers+lines',
                                         name='前起放下',
                                         marker=dict(symbol='x', size=10)))  # X形点

                fig.update_layout(title=f'{aircraft_model} 起落架收放时间趋势',
                                  xaxis_title='日期',
                                  yaxis_title='时间（秒）')

                st.plotly_chart(fig)

if __name__ == '__main__':
    main()
