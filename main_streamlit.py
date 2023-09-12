import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 设置页面布局
st.set_page_config(layout="wide", page_title="Data Analysis")

def clean_data(data, columns):
    # 转换格式并设置时间为索引
    data['Time'] = pd.to_datetime(data['Time'])
    # data.columns = data.columns.interpolate(method='linear')
    data.set_index('Time', inplace=True)
    data.sort_values('Time', inplace=True)
    return data

def calculate_difference(data1, column1, column2):
    data1[column1] = pd.to_numeric(data1[column1], errors='coerce')
    data1[column2] = pd.to_numeric(data1[column2], errors='coerce')
    data1[column1] = data1[column1].interpolate(method='linear')
    data1[column2] = data1[column2].interpolate(method='linear')
    data1['Difference'] = abs(data1[column1] - data1[column2])
    return data1

def calculate_quotient(data1, column1, column2):
    data1[column1] = pd.to_numeric(data1[column1], errors='coerce')
    data1[column2] = pd.to_numeric(data1[column2], errors='coerce')
    data1[column1] = data1[column1].interpolate(method='linear')
    data1[column2] = data1[column2].interpolate(method='linear')
    data1['Quotient'] = data1[column1] / data1[column2]
    return data1

def create_line_chart(data, column1, column2, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[column1], mode='lines', name=column1))
    fig.add_trace(go.Scatter(x=data.index, y=data[column2], mode='lines', name=column2))
    fig.update_layout(
        title=title,
        showlegend=True,
        width=1200,   # 设置画布宽度为1480像素
        height=600,   # 设置画布高度为800像素
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),   # 显示x轴网格虚线
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')    # 显示y轴网格虚线
    )
    # fig.update_yaxes(dtick=20)
    return fig

def main():

    st.title("译码数据可视化程序")

    # 创建一个输入框来获取header的值
    st.write("请输入数据表格中，列名位于第几行？手动译码数据输入0，自动译码数据则输入4或5：")
    header = st.text_input("Enter header value", "0")

    # 导入数据
    uploaded_file = st.file_uploader("请选择要导入的数据文件", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, header=int(header))
            st.success("数据已成功导入！")
        elif file_extension == "xlsx":
            data = pd.read_excel(uploaded_file, header=int(header))
        else:
            st.sidebar.warning("不支持的文件格式！")
            return

        # 数据清洗
        columns = data.columns
        data = clean_data(data, columns)
        # 显示表格数据的前10行
        st.write("表格数据的前5行：")
        st.table(data.head())

        # 添加自定义CSS样式
        st.markdown(
            """
            <style>
            table td:nth-child(odd) {
                background-color: #f2f2f2;
            }
            table td:nth-child(even) {
                background-color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # 选择列
        with st.sidebar:
            columns = st.multiselect("请选择要分析的列", data.columns)
        if len(columns) > 0:
            st.write(f"已选择的列：{', '.join(columns)}")

            # 使用Plotly绘制图表
            fig = px.line(data, x=data.index, y=columns, title="数据可视化")
            # 添加一个滑动条，实现在图表上进行缩放和选择日期范围
            fig.update_xaxes(rangeslider_visible=True)
            # 更新布局
            fig.update_layout(
                showlegend=True,
                width=1200,
                height=600,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
            # 显示图表
            st.plotly_chart(fig)

        else:
            with st.sidebar:
                if len(columns) == 0:
                    st.warning("请先选择要分析的列！")

        with st.sidebar:
            columns1 = st.multiselect("请选择要进一步分析的列", data.columns)

        if len(columns1) == 2:
            st.write(f"已选择的列：{', '.join(columns1)}")

            # 选择图表类型
            chart_type = st.selectbox("请选择计算方式", ["逐行求差", "逐行相除"])

            if st.button("生成图表"):
                if chart_type == "逐行求差":

                    # 计算差值并添加到数据中
                    data1 = calculate_difference(data, columns1[0], columns1[1])
                    fig = create_line_chart(data1, columns1[0], columns1[1], "数据差值可视化")
                    fig.add_trace(go.Scatter(x=data.index, y=abs(data1['Difference']), mode='lines', name='Absolute Difference'))
                    st.plotly_chart(fig)

                elif chart_type == "逐行相除":

                    # 计算比值并添加到数据中
                    data1 = calculate_quotient(data, columns1[0], columns1[1])
                    fig = create_line_chart(data1, columns1[0], columns1[1], "数据比值可视化")
                    fig.add_trace(go.Scatter(x=data.index, y=abs(data1['Quotient']), mode='lines', name='Ratio'))
                    st.plotly_chart(fig)

        elif len(columns) > 2:
            st.warning("只能选择两列进行计算！")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Copyright © 2023, 数据可视化   王康业")

if __name__ == "__main__":
    main()
