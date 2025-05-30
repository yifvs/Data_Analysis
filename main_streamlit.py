import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# 设置页面布局
st.set_page_config(layout="wide", page_title="Data Analysis", page_icon="📊")

def main():

    st.title(":blue[译码数据可视化程序] ✈")
    st.markdown("---")
    
    # 使用列布局优化界面
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 数据配置")
        # 创建一个输入框来获取header的值
        st.markdown("**列名行位置设置**")
        header = st.selectbox(
            "请选择数据表格中列名所在的行：",
            options=["0", "4"],
            index=1,
            format_func=lambda x: f"第{x}行 ({'手动译码数据' if x=='0' else '自动译码数据'})"
        )
        
    with col2:
        st.markdown("### 🗑️ 数据清理")
        # 添加两个输入框来获取要删除的行数
        st.markdown("**无效数据删除设置**")
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            num_rows_to_skip_before = st.number_input("前部删除行数", min_value=0, value=0, help="删除数据开头的无效行")
        with col2_2:
            num_rows_to_skip_after = st.number_input("尾部删除行数", min_value=0, value=0, help="删除数据末尾的无效行")
    
    colors = ['tomato', 'red', 'violet', 'cyan', 'orange', 'pink', 'brown', 'skyblue', 'white', 'olive', 'blue', 'forestgreen', 'cornflowerblue']
    
    st.markdown("---")
    st.markdown("### 📁 文件上传")
    # 导入数据
    uploaded_file = st.file_uploader(
        "请选择要导入的数据文件", 
        type=["csv", "xlsx"],
        help="支持CSV和Excel文件格式"
    )
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "csv":
            try:
                data = pd.read_csv(uploaded_file, index_col="Time", header=int(header), dtype='str', encoding='gb18030')
            except ValueError:
                data = pd.read_csv(uploaded_file, index_col="TIME", header=int(header), dtype='str', encoding='gb18030')
            st.success("数据已成功导入！")
        elif file_extension == "xlsx":
            try:
                data = pd.read_excel(uploaded_file, index_col="Time", header=int(header), dtype='str')
            except ValueError:
                data = pd.read_excel(uploaded_file, index_col="TIME", header=int(header), dtype='str')
        else:
            st.sidebar.warning("不支持的文件格式！")
            return
        
        # 删除前面指定的行数
        if num_rows_to_skip_before > 0:
            data = data.iloc[num_rows_to_skip_before:]

        # 删除后面指定的行数
        if num_rows_to_skip_after > 0:
            data = data.iloc[:-num_rows_to_skip_after]
            
        # 显示表格数据
        st.subheader("表格数据：")
        show_data = st.checkbox('是否显示表格数据', value=False)
        if show_data:
            st.dataframe(data)

        # 选择列
        with st.sidebar:
            string_columns = st.multiselect(":blue[请选择要分析的列（字符串类型参数）]", data.columns)
            numeric_columns = st.multiselect(":blue[请选择要分析的列（数值类型参数）]", data.columns)
            
            # 添加多子图显示选项
            multi_subplot_mode = st.checkbox(":green[启用多子图显示模式（每个参数独立Y轴）]", value=False)
            if multi_subplot_mode:
                st.info("多子图模式：每个参数将使用独立的Y轴，但共享X轴进行同步缩放")
                # 添加紧凑模式选项
                compact_mode = st.checkbox(":orange[启用紧凑模式（两列显示）]", value=False)
                if compact_mode:
                    st.info("紧凑模式：子图将分为两列显示，节省垂直空间")
            else:
                compact_mode = False
            
        # 检查是否选择了任何列
        if len(string_columns) > 0 or len(numeric_columns) > 0:
            # 合并所有选择的列
            all_selected_columns = string_columns + numeric_columns
            
            # 多子图显示模式
            if multi_subplot_mode and len(all_selected_columns) > 0:
                st.write(f"多子图模式 - 已选择的列：{', '.join(all_selected_columns)}")
                
                # 数据预处理
                for column in string_columns:
                    data[column] = data[column].astype(str)
                
                for column in numeric_columns:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                    data[column] = data[column].interpolate(method='linear')
                
                # 创建多子图布局（每个参数一个子图）
                subplot_count = len(all_selected_columns)
                if compact_mode:
                    # 紧凑模式：两列布局
                    rows = (subplot_count + 1) // 2  # 向上取整
                    cols = 2 if subplot_count > 1 else 1
                    fig = make_subplots(
                        rows=rows, cols=cols,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        horizontal_spacing=0.05
                    )
                else:
                    # 标准模式：单列布局
                    fig = make_subplots(
                        rows=subplot_count, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02
                    )
                
                # 为每个参数添加子图
                for i, column in enumerate(all_selected_columns):
                    if compact_mode:
                        # 紧凑模式：计算行列位置
                        row_num = (i // 2) + 1
                        col_num = (i % 2) + 1
                    else:
                        # 标准模式：单列布局
                        row_num = i + 1
                        col_num = 1
                    
                    if column in string_columns:
                        # 字符串类型数据转换为数值，但保留原始值用于悬停
                        string_values = [hash(str(val)) % 1000 for val in data[column]]
                        original_strings = [str(val) for val in data[column]]
                        fig.add_trace(
                            go.Scatter(
                                x=data.index, 
                                y=string_values, 
                                mode='lines',
                                name=column,
                                line=dict(color=colors[i % len(colors)], width=2),
                                customdata=original_strings,
                                hovertemplate=f'{column}: %{{customdata}}<br>Hash值: %{{y}}<extra></extra>'
                            ),
                            row=row_num, col=col_num
                        )
                        # 设置Y轴标题
                        fig.update_yaxes(
                            title_text=f"{column}",
                            showgrid=True, gridwidth=1, gridcolor='lightgray',
                            showline=True, linewidth=1, linecolor='black',
                            row=row_num, col=col_num
                        )
                    else:
                        # 数值类型数据
                        fig.add_trace(
                            go.Scatter(
                                x=data.index, 
                                y=data[column], 
                                mode='lines',
                                name=column,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate=f'{column}: %{{y}}<extra></extra>'
                            ),
                            row=row_num, col=col_num
                        )
                        # 设置Y轴标题
                        fig.update_yaxes(
                            title_text=column,
                            showgrid=True, gridwidth=1, gridcolor='lightgray',
                            showline=True, linewidth=1, linecolor='black',
                            row=row_num, col=col_num
                        )
                
                # 为每个数据点的悬停标签设置个性化的背景颜色
                for i in range(len(fig.data)):
                    fig.data[i].hoverlabel = dict(
                        bgcolor=colors[i % len(colors)], 
                        font=dict(size=12, color='white', family='Arial')
                    )
                
                # 更新X轴（只在最底部显示标签和滑动条）
                if compact_mode:
                    rows = (subplot_count + 1) // 2
                    cols = 2 if subplot_count > 1 else 1
                    for i in range(subplot_count):
                        row_num = (i // 2) + 1
                        col_num = (i % 2) + 1
                        if row_num == rows:  # 最后一行的子图
                            fig.update_xaxes(
                                showgrid=True, gridwidth=1, gridcolor='lightgray',
                                showline=True, linewidth=1, linecolor='black',
                                tickmode='linear', dtick=300, tickangle=45,
                                rangeslider=dict(visible=True, thickness=0.1),
                                title_text="时间",
                                row=row_num, col=col_num
                            )
                        else:
                            fig.update_xaxes(
                                showgrid=True, gridwidth=1, gridcolor='lightgray',
                                showline=True, linewidth=1, linecolor='black',
                                tickmode='linear', dtick=300,
                                showticklabels=False,
                                row=row_num, col=col_num
                            )
                else:
                    for i in range(subplot_count):
                        row_num = i + 1
                        if row_num == subplot_count:  # 最后一个子图
                            fig.update_xaxes(
                                showgrid=True, gridwidth=1, gridcolor='lightgray',
                                showline=True, linewidth=1, linecolor='black',
                                tickmode='linear', dtick=300, tickangle=45,
                                rangeslider=dict(visible=True, thickness=0.1),
                                title_text="时间",
                                row=row_num, col=1
                            )
                        else:
                            fig.update_xaxes(
                                showgrid=True, gridwidth=1, gridcolor='lightgray',
                                showline=True, linewidth=1, linecolor='black',
                                tickmode='linear', dtick=300,
                                showticklabels=False,  # 隐藏中间子图的X轴标签
                                row=row_num, col=1
                            )
                
                # 更新整体布局
                if compact_mode:
                    rows = (subplot_count + 1) // 2
                    height = 300 * rows + 100
                    title = "多子图模式（紧凑布局） - 每个参数独立Y轴"
                else:
                    height = 200 * subplot_count + 100
                    title = "多子图模式 - 每个参数独立Y轴"
                
                fig.update_layout(
                    showlegend=True, 
                    width=1200, 
                    height=height,
                    hovermode='x',  # 改为'x'模式以实现更好的联动
                    title=title,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig)
                
            # 原有的显示模式
            elif not multi_subplot_mode:
                # 创建子图布局
                subplot_count = 0
                if len(string_columns) > 0:
                    subplot_count += 1
                if len(numeric_columns) > 0:
                    subplot_count += 1
                
            if subplot_count == 1:
                # 只有一种类型的数据
                if len(string_columns) > 0:
                    # 只有字符串类型数据
                    st.write(f"已选择的字符串列：{', '.join(string_columns)}")
                    for column in string_columns:
                        data[column] = data[column].astype(str)
                    
                    fig = px.line(data, x=data.index, y=string_columns, title="字符串类型数据可视化", line_shape='linear')
                    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.1))
                    fig.update_layout(
                        showlegend=True, width=1200, height=600,
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black', tickmode='linear', dtick=300),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig)
                    
                elif len(numeric_columns) > 0:
                    # 只有数值类型数据
                    st.write(f"已选择的数值列：{', '.join(numeric_columns)}")
                    selected_columns = data.columns
                    for column in selected_columns:
                        data[column] = pd.to_numeric(data[column], errors='coerce')
                        data[column].interpolate(method='linear', inplace=True)
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    secondary_axis = st.selectbox(":blue[请选择作为副轴的列（如果有的话）]", options=[None] + numeric_columns)
                    primary_axis_columns = list(set(numeric_columns) - set([secondary_axis])) if secondary_axis else numeric_columns
                    
                    for column in primary_axis_columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column, line=dict(width=2)),secondary_y=False)
                    
                    if secondary_axis:
                        fig.add_trace(go.Scatter(x=data.index, y=data[secondary_axis], mode='lines', name=secondary_axis, line=dict(width=1)), secondary_y=True)
                    
                    for i in range(len(fig.data)):
                        fig.data[i].hoverlabel = dict(bgcolor=colors[i % len(colors)], font=dict(size=14, color='black', family='Arial'))
                    
                    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.1))
                    fig.update_layout(
                        showlegend=True, width=1200, height=600, hovermode='x',
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot', showline=True, linewidth=1, linecolor='black', tickmode='linear', dtick=300),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot', showline=True, linewidth=1, linecolor='black'),
                        yaxis2=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot', showline=True, linewidth=1, linecolor='black', overlaying='y', side='right'),
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig)
                    
            else:
                # 两种类型的数据都有，创建共享X轴的子图
                st.write(f"已选择的字符串列：{', '.join(string_columns)}")
                st.write(f"已选择的数值列：{', '.join(numeric_columns)}")
                
                # 数据预处理
                for column in string_columns:
                    data[column] = data[column].astype(str)
                
                selected_columns = data.columns
                for column in selected_columns:
                    if column in numeric_columns:
                        data[column] = pd.to_numeric(data[column], errors='coerce')
                        data[column] = data[column].interpolate(method='linear')
                
                # 创建共享X轴的子图
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    # subplot_titles=("字符串类型参数", "数值类型参数"),
                    specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
                )
                
                # 添加字符串类型数据到第一个子图
                for i, column in enumerate(string_columns):
                    # 将字符串转换为数值以便绘图（使用hash值），但保留原始值用于悬停
                    string_values = [hash(str(val)) % 1000 for val in data[column]]
                    original_strings = [str(val) for val in data[column]]
                    fig.add_trace(
                        go.Scatter(x=data.index, y=string_values, mode='lines', name=f"字符串-{column}", line=dict(color=colors[i % len(colors)], width=2), customdata=original_strings, hovertemplate=f'{column}: %{{customdata}}<br>Hash值: %{{y}}<extra></extra>'),
                        row=1, col=1
                    )
                
                # 添加数值类型数据到第二个子图
                secondary_axis = st.selectbox(":blue[请选择作为副轴的列（如果有的话）]", options=[None] + numeric_columns)
                primary_axis_columns = list(set(numeric_columns) - set([secondary_axis])) if secondary_axis else numeric_columns
                
                for i, column in enumerate(primary_axis_columns):
                    fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=f"数值-{column}", line=dict(color=colors[(i + len(string_columns)) % len(colors)], width=2), hovertemplate=f'{column}: %{{y}}<extra></extra>'),
                        row=2, col=1, secondary_y=False
                    )
                
                if secondary_axis:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data[secondary_axis], mode='lines', name=f"数值副轴-{secondary_axis}", line=dict(color=colors[(len(primary_axis_columns) + len(string_columns)) % len(colors)], width=2), hovertemplate=f'{secondary_axis}: %{{y}}<extra></extra>'),
                        row=2, col=1, secondary_y=True
                    )
                
                # 为每个数据点的悬停标签设置个性化的背景颜色
                for i in range(len(fig.data)):
                    fig.data[i].hoverlabel = dict(bgcolor=colors[i % len(colors)], font=dict(size=14, color='black', family='Arial'))
                
                # 更新布局
                fig.update_layout(
                    showlegend=True, width=1200, height=800,
                    hovermode='x unified',
                    title="同步X轴的多类型数据可视化"
                )
                
                # 更新X轴（只需要更新底部的X轴，因为是共享的）
                fig.update_xaxes(
                    showgrid=True, gridwidth=1, gridcolor='lightgray', 
                    showline=True, linewidth=1, linecolor='black', 
                    tickmode='linear', dtick=300, tickangle=45,
                    rangeslider=dict(visible=True, thickness=0.1), row=2, col=1
                )
                
                # 更新Y轴
                fig.update_yaxes(
                    showgrid=True, gridwidth=1, gridcolor='lightgray',
                    showline=True, linewidth=1, linecolor='black',
                    title="字符串值（Hash）", row=1, col=1
                )
                
                fig.update_yaxes(
                    showgrid=True, gridwidth=1, gridcolor='lightgray',
                    showline=True, linewidth=1, linecolor='black',
                    title="数值", row=2, col=1, secondary_y=False
                )
                
                if secondary_axis:
                    fig.update_yaxes(
                        showgrid=True, gridwidth=1, gridcolor='lightgray',
                        showline=True, linewidth=1, linecolor='black',
                        title=f"副轴-{secondary_axis}", row=2, col=1, secondary_y=True
                    )
                
                st.plotly_chart(fig)
        else:
            with st.sidebar:
                st.warning("请先选择要分析的列！")

        st.sidebar.markdown("---")

        with st.sidebar:
            st.caption("自定义X轴和Y轴并生成散点图：")           
            x_column = st.selectbox(":blue[请选择X轴:]", options=[None]+data.columns.tolist())
            y_columns = st.multiselect(":blue[请选择Y轴(可多选):]", data.columns)
        if x_column and y_columns:
            st.write(f"已选择的列：{x_column}, {', '.join(y_columns)}")
            selected_data = data[[x_column] + y_columns]
            selected_data[x_column] = pd.to_numeric(selected_data[x_column], errors='coerce')  
            selected_data[x_column].interpolate(method='linear', inplace=True)  
            for column in y_columns:
                selected_data[column] = pd.to_numeric(selected_data[column], errors='coerce')  
                selected_data[column].interpolate(method='linear', inplace=True)  
            fig = go.Figure()
            for column in y_columns:
                fig.add_trace(go.Scatter(x=selected_data[x_column], y=selected_data[column], mode='markers', name=column))
            fig.update_xaxes(title=x_column)
            # fig.update_yaxes(title=y_columns)
            fig.update_layout(
                showlegend=True, width=1200, height=600,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                            linecolor='black', tickmode='linear', dtick=5),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1,
                            linecolor='black'),
                xaxis_tickangle=45
            )
            st.plotly_chart(fig)
        else:
            with st.sidebar:
                st.warning("请先选择要自定义的X轴和Y轴！")

        st.sidebar.markdown("---")

        with st.sidebar:
            columns1 = st.multiselect(":blue[请选择需要计算列]", data.columns)

        if len(columns1) >= 2:
            st.write(f"已选择的列：{', '.join(columns1)}")
            # 在侧边栏添加5个文本输入框，允许用户输入运算公式
            formulas = []
            for i in range(5):
                formula = st.sidebar.text_input(f"输入运算公式{i + 1}（使用列名变量）")
                formulas.append(formula)
            dtick_value = st.sidebar.text_input(":violet[请输入副轴Y2的刻度间隔值(不输入，则默认间隔为10)：]")
            # 添加一个提交按钮
            if st.sidebar.button("Submit"):
                selected_columns = data.columns
                for column in selected_columns:
                    data[column] = pd.to_numeric(data[column], errors='coerce')  # 转换为数字类型
                    data[column].interpolate(method='linear', inplace=True)  # 使用线性插值填充空值
                try:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    for column in columns1:
                        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column, line=dict(width=2)),secondary_y=False)
                    for i, formula in enumerate(formulas):
                        if formula:
                            # 使用eval函数计算公式并将结果添加为新列
                            data[f'计算结果{i + 1}'] = data.eval(formula.replace('//', '/'))
                            # 将新列的曲线添加到图表中
                            fig.add_trace(go.Scatter(x=data.index, y=data[f'计算结果{i + 1}'], mode='lines', name=f'{formula}', line=dict(width=2)), secondary_y=True)
                  # 为每个数据点的悬停标签设置个性化的背景颜色  
                    for i in range(len(fig.data)):
                        fig.data[i].hoverlabel = dict(bgcolor=colors[i], font=dict(size=14, color='black', family='Arial'))
                        
                    if dtick_value:
                        dtick_value = float(dtick_value)
                    else:
                        dtick_value = 10
                        
                    fig.update_layout(
                        showlegend=True, width=1200, height=600,
                        hovermode='x unified',
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black', tickmode='linear', dtick=300),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
                        yaxis2=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black', overlaying='y', side='right', dtick=dtick_value),
                        xaxis_tickangle=45
                    )
                    # 设置Y轴刻度对齐
                    fig.update_yaxes(matches='y')
                    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.1))
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"运算出错：{str(e)}")
                        
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 版权声明        
    Copyright © 2024 海航航空技术有限公司. All Rights Reserved.                          
    本应用程序受著作权法和其他知识产权法保护。未经授权，禁止复制、修改或分发本程序的任何部分。
    """)
    st.sidebar.markdown("Report Bug : kangy_wang@hnair.com")
    # 添加一些空行来确保版权信息在底部
    st.sidebar.markdown("<br>" * 5, unsafe_allow_html=True)

# # 接受一个参数 x，并检查该参数是否为字符串类型。如果是字符串类型，则使用正则表达式提取小数部分，并返回第一个匹配的小数
# def extract_number(x):
#     if isinstance(x, str):
#         numbers = re.findall('\d+\.\d+', x)
#         if numbers:
#             return round(float(numbers[0]), 2)    # 使用 round 函数将其精确到小数点后两位
#     return x   # 如果不是字符串类型，则直接返回原始值

if __name__ == "__main__":
    main()
