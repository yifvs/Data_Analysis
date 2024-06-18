import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 设置页面布局
st.set_page_config(layout="wide", page_title="批量数据分析", page_icon="📊")

def main():

    st.title(":blue[批量数据分析] ✈")
    st.write(":violet[本页面主要用于批量读取译码数据，对比不同航段，单个参数的变化趋势]")

    # 上传文件
    uploaded_files = st.file_uploader("📁 上传文件", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        # 保存所有文件的数据框
        data_frames = []
        file_names = []
        # 处理每个上传的文件
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_names.append(file_name)
            # 读取文件
            df = pd.read_csv(uploaded_file, header=4, index_col="Time", encoding='gb18030', low_memory=False)
            data_frames.append(df)   # 每次读取一个上传的文件后，将该文件的数据添加到data_frames列表中

        # 在侧边栏中创建一个下拉菜单，用于选择要分析的列
        with st.sidebar:
            # 获取所有文件的列名
            columns = []
            for df in data_frames:
                columns.extend(df.columns.tolist())
            columns = list(set(columns))  # 去重
            selected_columns = st.multiselect("请选择要分析的列（目前只支持数值类型参数）", columns)
            selected_flight_phases = st.multiselect("请选择要分析的FLIGHT_PHASE", ["INTER FLT", "ENG START", "TAXI OUT", "TAKE OFF", "INIT CLIMB", "CLIMB", "CRUISE", "DESCENT", "APPROACH", "FINAL APP", "FLARE", "LANDING", "TAXI IN"])
            st.write(f"已选择的列：{', '.join(selected_columns)}")
            selected_filter = st.multiselect("是否增加其它筛选条件", columns)

            filter_conditions = {}
            for filter_option in selected_filter:
                filter_formula = st.text_input(f"输入筛选公式 ({filter_option})", help=f"例如：{filter_option}>60")
                filter_conditions[filter_option] = filter_formula

            generate_chart_button = st.button("生成图表")
            selected_metric = st.selectbox("选择要分析的指标:", ["最大值", "最小值", "平均值", "方差"], index=0)
            data_analysis_button = st.button("数据分析")

        # 对所有数据框进行处理
        if generate_chart_button and selected_columns:
            for i, df in enumerate(data_frames):
                # 数据清洗
                for column in selected_columns:
                    if column in df.columns:
                        df[column] = pd.to_numeric(df[column], errors='coerce')  # 转换为数字类型
                        df[column].interpolate(method='linear', inplace=True)  # 使用线性插值填充空值
                
                # 应用附加的筛选条件
                for filter_option, filter_formula in filter_conditions.items():
                    if filter_formula:
                        try:
                            df[filter_option] = pd.to_numeric(df[filter_option], errors='coerce')  # 转换为数字类型
                            df = df.query(filter_formula)
                        except Exception as e:
                            st.error(f"筛选公式错误 ({filter_option}): {filter_formula}")
                            st.error(e)
                            return

                # 创建Plotly图表对象
                fig = go.Figure()
                for column in selected_columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))
                fig.update_layout(
                    title=f"{', '.join(selected_columns)} - {file_names[i]}",
                    showlegend=True, width=800, height=400,
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                               linecolor='black', tickmode='linear', dtick=300),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                               linecolor='black'),
                    xaxis_tickangle=45
                )
                # 根据选择的FLIGHT_PHASE进行数据切分
                if selected_flight_phases:
                    for phase in selected_flight_phases:
                        phase_df = df[df['FLIGHT_PHASE'] == phase]
                        for column in selected_columns:
                            fig.add_trace(go.Scatter(x=phase_df.index, y=phase_df[column], mode='lines', name=f"{phase} - {column}", line=dict(color="#ff0000")))

                # 显示图表在页面中
                st.plotly_chart(fig)

        if data_analysis_button and selected_columns and selected_metric:
            st.write(f"已选择的指标：{selected_metric}")

            df_results = pd.DataFrame(columns=selected_columns)  # 初始化DataFrame

            for i, df in enumerate(data_frames):
                for column in selected_columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')  # 转换为数字类型
                    df[column].interpolate(method='linear', inplace=True)  # 使用线性插值填充空值

                # 应用附加的筛选条件
                for filter_option, filter_formula in filter_conditions.items():
                    if filter_formula:
                        try:
                            df[filter_option] = pd.to_numeric(df[filter_option], errors='coerce')  # 转换为数字类型
                            df = df.query(filter_formula)
                        except Exception as e:
                            st.error(f"筛选公式错误 ({filter_option}): {filter_formula}")
                            st.error(e)
                            return

                for phase in selected_flight_phases:
                    phase_df = df[df['FLIGHT_PHASE'] == phase]
                    for column in selected_columns:
                        result = calculate_metric(phase_df[column], selected_metric)
                        df_results.loc[f"{file_names[i]} - {phase}", column] = result

            # 创建两个列，比例为1:1
            col1, col2 = st.columns(2)

            with col1:
                st.write(df_results)

            with col2:
                # 创建Plotly图表对象
                fig = go.Figure()
                for column in df_results.columns:
                    # 提取文件名和航段作为横坐标
                    x_values = df_results.index.tolist()
                    fig.add_trace(
                        go.Scatter(x=x_values, y=df_results[column],
                                   mode='markers', name=column, marker=dict(size=10))
                    )
                fig.update_layout(
                    title=f"数据分析 - 散点图",
                    showlegend=True, width=800, height=400,
                    xaxis=dict(title='样本', showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True,
                               linewidth=1,
                               linecolor='black'),
                    yaxis=dict(title=selected_metric, showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True,
                               linewidth=1,
                               linecolor='black'),
                    xaxis_tickangle=45
                )
                # 显示图表在页面中
                st.plotly_chart(fig)

def calculate_metric(series, metric):
    if metric == "最大值":
        return series.max()
    elif metric == "最小值":
        return series.min()
    elif metric == "平均值":
        return series.mean()
    elif metric == "方差":
        return series.var()

if __name__ == "__main__":
    main()
