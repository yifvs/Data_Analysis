import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# 从环境变量中读取OpenAI API密钥
api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("请设置OPENAI_API_KEY环境变量")
    st.stop()

# Streamlit应用程序标题
st.title("Chat with Excel")
st.info(":violet[This app is still in beta. Please report any bugs to kangy_wang@hnair.com]", icon='⚠')

# 用户输入header值
st.write("请输入数据表格中，列名位于第几行？手动译码数据输入0，自动译码数据则输入4或5：")
header = st.number_input("Enter header value", min_value=0, value=0)

# 用户输入要删除的行数
st.write("是否要删除无效的干扰数据？请分别在以下两个输入框内输入相应的行数：")
num_rows_to_skip_before = st.number_input("Number of rows to skip（FWD）：", min_value=0, value=0)
num_rows_to_skip_after = st.number_input("Number of rows to discard（AFT）：", min_value=0, value=0)

# 文件上传器
uploaded_file = st.file_uploader("📁 请选择要导入的数据文件", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    try:
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, header=int(header), encoding='gb18030')
        elif file_extension == "xlsx":
            data = pd.read_excel(uploaded_file, header=int(header))
        else:
            st.warning("不支持的文件格式！")
            data = None
    except Exception as e:
        st.error(f"文件读取错误: {e}")
        data = None

    if data is not None:
        # 数据预处理：删除指定行数
        if num_rows_to_skip_before > 0:
            data = data.iloc[num_rows_to_skip_before:]
        if num_rows_to_skip_after > 0:
            data = data.iloc[:-num_rows_to_skip_after]

        # 使用线性插值填充空值
        data = data.interpolate(method='linear')

        st.write("数据预览：")
        st.dataframe(data.head())  # 使用st.dataframe来显示DataFrame

        # 初始化PandasAI和OpenAI
        try:
            llm = OpenAI(api_token=api_key)
            pandas_ai = PandasAI(llm)
        except Exception as e:
            st.error(f"初始化错误: {e}")
            pandas_ai = None

        # 用户输入自然语言查询
        user_query = st.text_input("请输入你的问题：")

        if st.button("提交"):
            if user_query and pandas_ai:
                try:
                    # 使用PandasAI进行自然语言处理和数据分析
                    response = pandas_ai.run(data, prompt=user_query)
                    st.write("查询结果：")
                    st.write(response)
                except Exception as e:
                    st.error(f"查询过程中出错: {e}")
            else:
                st.warning("请输入一个查询问题。")

else:
    st.write("请上传一个CSV或Excel文件以开始数据分析。")
