import streamlit as st
import pandas as pd
from glob import glob
import plotly.graph_objects as go

def main():

    st.title(":blue[批量数据分析] ✈")
    st.write(":violet[本页面主要用于批量读取译码数据，对比不同航段，同一参数的变化情况。后续会开发趋势功能，敬请期待]")
    # 创建一个输入框来获取header的值
    st.write("请输入数据表格中，列名位于第几行？手动译码数据输入0，自动译码数据则输入4或5：")
    header = st.text_input("Enter header value", "0")

    # 创建一个文本输入框，提示用户输入文件夹路径
    file_folder = st.text_input("请输入译码数据文件夹路径：")
    submit_button = st.button("Submit")

    # 当用户输入路径后，将其赋值给file_path变量
    if submit_button:
        file_path = file_folder.strip()
        file_list = glob(file_path + '/*.csv')

        if file_list:
            # 读取第一个文件，获取列名
            data = pd.read_csv(file_list[0], header=int(header), index_col="Time", encoding='gb18030')
            columns = data.columns.tolist()
            # 在侧边栏中创建一个下拉菜单，用于选择要分析的列
            with st.sidebar:
                selected_columns = st.multiselect("请选择要分析的列（目前只支持数值类型参数）", [""] + columns)
                generate_chart_button = st.button("生成图表")

            for file in file_list:
                data = pd.read_csv(file, header=int(header), index_col="Time", encoding='gb18030')
                time = pd.read_csv(file, usecols=[0], squeeze=True, encoding='gb18030')
                time = time[1][22:41]
                time = pd.to_datetime(time, dayfirst=True, errors='coerce')

                if generate_chart_button and selected_columns:
                    st.write(f"已选择的列：{', '.join(selected_columns)}")
                    for column in selected_columns:
                        if column in data.columns:
                            data[column] = pd.to_numeric(data[column], errors='coerce')  # 转换为数字类型
                            data[column].interpolate(method='linear', inplace=True)  # 使用线性插值填充空值
                        else:
                            st.write(f"列名 {column} 不存在于数据中")
                    # 创建Plotly图表对象
                    fig = go.Figure()
                    for column in selected_columns:
                        if column in data.columns:
                            fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column))
                    fig.update_layout(
                        title=f"{', '.join(selected_columns)} - {time}",
                        showlegend=True,
                        width=800,
                        height=400,
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                                    linecolor='black', tickmode='linear', dtick=300),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                                    linecolor='black'),
                        xaxis_tickangle=45
                        )
                    # 显示图表在页面中
                    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
