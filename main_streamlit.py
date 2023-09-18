import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 设置页面布局
st.set_page_config(layout="wide", page_title="Data Analysis")

def main():

    st.title(":blue[译码数据可视化程序] ✈")

    # 创建一个输入框来获取header的值
    st.write("请输入数据表格中，列名位于第几行？手动译码数据输入0，自动译码数据则输入4或5：")
    header = st.text_input("Enter header value", "0")

    # 导入数据
    uploaded_file = st.file_uploader("请选择要导入的数据文件", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, index_col="Time", header=int(header), encoding='gb18030')
            st.success("数据已成功导入！")
        elif file_extension == "xlsx":
            data = pd.read_excel(uploaded_file, index_col="Time", header=int(header), encoding='gb18030')
        else:
            st.sidebar.warning("不支持的文件格式！")
            return

        # 显示表格数据的前10行
        st.write("表格数据的前5行：")
        st.dataframe(data.head())

        # 选择列
        with st.sidebar:
            columns = st.multiselect(":blue[请选择要分析的列（字符串类型参数）]", data.columns)
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
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),   
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),   
                xaxis_tickangle=45
            )
            # 显示图表
            st.plotly_chart(fig)

        else:
            with st.sidebar:
                if len(columns) == 0:
                    st.warning("请先选择要分析的列！")

        with st.sidebar:
            columns = st.multiselect(":blue[请选择要分析的列（数值类型参数）]", data.columns)
        if len(columns) > 0:
            st.write(f"已选择的列：{', '.join(columns)}")
            selected_columns = data.columns
            for column in selected_columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')  # 转换为数字类型
                data[column].interpolate(method='linear', inplace=True)  # 使用线性插值填充空值
            # 使用Plotly绘制图表
            fig = px.line(data, x=data.index, y=columns, title="数据可视化")
            # 添加一个滑动条，实现在图表上进行缩放和选择日期范围
            fig.update_xaxes(rangeslider_visible=True)
            # 更新布局
            fig.update_layout(
                showlegend=True,
                width=1200,
                height=600,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
                xaxis_tickangle=45
            )
            # 显示图表
            st.plotly_chart(fig)

        else:
            with st.sidebar:
                if len(columns) == 0:
                    st.warning("请先选择要分析的列！")

        
        with st.sidebar:
            columns1 = st.multiselect("请选择需要进行计算列", data.columns)

        if len(columns1) >= 2:
            st.write(f"已选择的列：{', '.join(columns1)}")
            # 在侧边栏添加一个文本输入框，允许用户输入运算公式
            formula = st.sidebar.text_input(":red[输入运算公式（使用列名变量）]")
            # 添加一个提交按钮
            if st.sidebar.button("Submit"):
                if formula:
                    selected_columns = data.columns
                    for column in selected_columns:
                        data[column] = pd.to_numeric(data[column], errors='coerce')  # 转换为数字类型
                        data[column].interpolate(method='linear', inplace=True)  # 使用线性插值填充空值
                    try:
                        # 使用eval函数计算公式并将结果添加为新列
                        data['计算结果'] = abs(data.eval(formula.replace('//', '/')))

                        # 使用Plotly创建图表
                        fig = px.line(data, x=data.index, y=columns1, title='计算结果图表')
                        fig.add_trace(go.Scatter(x=data.index, y=data['计算结果'], mode='lines', name='计算结果'))
                        fig.update_layout(
                            showlegend=True,
                            width=1200,
                            height=600,
                            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                                       linecolor='black'),
                            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                                       linecolor='black'),
                            xaxis_tickangle=45
                        )
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"运算出错：{str(e)}")


    st.sidebar.markdown("---")
    st.sidebar.markdown("Copyright © 2023, 数据可视化")
    st.sidebar.markdown("Co-operator: 黄栋梁")
    st.sidebar.markdown("Developer: 王康业")
    st.sidebar.markdown("Report Bug : kangy_wang@hnair.com")

if __name__ == "__main__":
    main()
