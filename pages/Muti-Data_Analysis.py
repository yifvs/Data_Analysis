import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def main():

    st.title(":blue[批量数据分析] ✈")
    st.write(":violet[本页面主要用于批量读取译码数据，对比不同航段，同一参数的变化情况。]")
    st.write("请输入数据表格中，列名位于第几行？手动译码数据输入0，自动译码数据则输入4或5：")
    header = st.text_input("Enter header value", "4")

    # 上传文件
    uploaded_files = st.file_uploader("上传文件", type=["csv"], accept_multiple_files=True)

    if uploaded_files is not None:
        # 保存所有文件的数据框
        data_frames = []
        file_names = []
        # 处理每个上传的文件
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_names.append(file_name)
            # 读取文件
            df = pd.read_csv(uploaded_file, header=int(header), index_col="Time", encoding='gb18030')
            data_frames.append(df)

        # 在侧边栏中创建一个下拉菜单，用于选择要分析的列
        with st.sidebar:
            # 获取所有文件的列名
            columns = []
            for df in data_frames:
                columns.extend(df.columns.tolist())
            selected_columns = st.multiselect("请选择要分析的列（目前只支持数值类型参数）", [""] + columns)
            generate_chart_button = st.button("生成图表")
            st.write(f"已选择的列：{', '.join(selected_columns)}")
        # 对所有数据框进行处理
        for i, df in enumerate(data_frames):
            if generate_chart_button and selected_columns:
                # 数据清洗
                for column in selected_columns:
                    if column in df.columns:
                        df[column] = pd.to_numeric(df[column], errors='coerce')  # 转换为数字类型
                        df[column].interpolate(method='linear', inplace=True)  # 使用线性插值填充空值
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

                # 显示图表在页面中
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
