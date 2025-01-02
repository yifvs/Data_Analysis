import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re

# 设置页面布局为宽模式
st.set_page_config(layout="wide", page_title='液压油量变化趋势', page_icon='🛫')

# 正则表达式匹配 YYYYMMDD_HHMMSS 格式
pattern = re.compile(r"(\d{8}_\d{6})")

@st.cache_data
def clean_data(df, columns):
    df = df[columns]
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    # 向后插值
    df.interpolate(method='bfill', inplace=True)
    # 向前插值
    df.interpolate(method='ffill', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df

@st.cache_data
def process_file(file, columns):
    file_name = file.name  # 提取文件名
    match = pattern.search(file_name)  # 匹配日期时间部分
    if match:
        long_string = match.group(1)
        try:
            time = datetime.strptime(long_string, "%Y%m%d_%H%M%S")
        except ValueError as e:
            st.error(f"Error parsing date from {long_string}: {e}")
            return None
    else:
        st.error(f"Filename {file_name} does not contain a valid date string.")
        return None
    
    # 读取CSV文件，跳过前4行作为header
    df = pd.read_csv(file, header=4, encoding='gb18030')
    df = clean_data(df, columns)  # 数据清洗
    if not df.empty:
        sysA_start = df[columns[2]].iloc[20]      
        sysA_end = df[columns[2]].iloc[-5]      
        sysB_start = df[columns[3]].iloc[20]
        sysB_end = df[columns[3]].iloc[-5]
        tat = df[columns[1]].iloc[5]
        df_result = pd.DataFrame({
            'time': [time],
            'TAT': [tat],
            'sysA起飞油量': [sysA_start], 
            'sysA落地油量': [sysA_end], 
            'sysB起飞油量': [sysB_start], 
            'sysB落地油量': [sysB_end], 
        })
        # 按时间排序
        df_result.sort_values(by=['time'], ascending=True, inplace=True)
        return df_result
    return None

st.title(":blue[液压系统油量变化趋势分析]")
st.info("受云服务器算力资源限制，上传太大的数据会导致网页崩溃。若有大量数据计算需求，请联系kangy_wang@hnair.com打开本地服务器使用")
aircraft_model = st.selectbox(":violet[请选择机型]", ("737NG", "737MAX"))

uploaded_files = st.file_uploader(":red[📁 请同时选中并拖拽多个CSV文档批量上传自动译码文件]", type=["csv"], accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) > 0:
    all_data = pd.DataFrame()

    # 根据选择的机型确定列名
    if aircraft_model == "737NG":
        columns = ['Time', 'TAT', 'OIL_QTY_HYDA', 'OIL_QTY_HYDB']
    elif aircraft_model == "737MAX":
        columns = ['Time', 'TAT1', 'HOQSYSA', 'HOQSYSB']

    for uploaded_file in uploaded_files:
        file_data = process_file(uploaded_file, columns)

        # 在合并 file_data 到 all_data 之前，检查 file_data 是否为 None，并确保它包含 time 列
        if file_data is not None and 'time' in file_data.columns:
            all_data = pd.concat([all_data, file_data], ignore_index=True)

    if not all_data.empty:
        # 按时间排序
        all_data.sort_values(by=['time'], ascending=True, inplace=True)
        # 时间范围选择器
        min_date = all_data['time'].min()
        max_date = all_data['time'].max()
        selected_dates = st.date_input(
            ":calendar: 选择时间范围",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # 过滤数据
        filtered_data = all_data[(all_data['time'] >= pd.Timestamp(selected_dates[0])) & (all_data['time'] <= pd.Timestamp(selected_dates[1]))]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('数据表格')
            st.write(filtered_data)

        with col2:
            st.subheader('趋势图')
            fig = px.line(filtered_data, x='time', y=['sysA起飞油量', 'sysA落地油量', 'sysB起飞油量', 'sysB落地油量'],
                          title='液压系统油量变化趋势',
                          labels={'value': 'HYD_QTY'},
                          markers=True,
                          template='plotly_white'
                        )
            
            fig.update_xaxes(title_text='Time', showgrid=True, gridwidth=1, gridcolor='LightPink', griddash='dash', rangeslider_visible=True)
            fig.update_yaxes(title_text='HYD_QTY', showgrid=True, gridwidth=1, gridcolor='LightBlue', griddash='dash')
            fig.update_layout(legend=dict(x=0, y=1, traceorder="normal", title="Legend"))
            
            st.plotly_chart(fig)
    else:
        st.warning("未成功提取任何数据，请检查上传文件的内容。")
else:
    st.info("请上传CSV文件以进行分析。")



