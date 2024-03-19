from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import pandas as pd
import streamlit as st

# 调整Streamlit页面的宽度
st.set_page_config(page_title="在Streamlit中使用PyGWalker", layout="wide", page_icon="📊")
# 在pygwalker和streamlit之间建立通信
init_streamlit_comm()

st.title("使用PyGWalker可视化表格数据")
# 获取一个pygwalker渲染器的实例
@st.cache_resource
def get_pyg_renderer(data) -> "StreamlitRenderer":
    # debug参数设置为False，防止其他用户向你的图表配置文件中写入数据
    return StreamlitRenderer(data, spec="./gw_config.json", debug=False)


# 导入数据
uploaded_file = st.file_uploader("请选择要导入的数据文件", type=["csv", "xls", "xlsx"],
                                 help="Scanned documents are not supported yet!")
if uploaded_file is not None:
    # 获取上传文件的扩展名
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension in ["csv", "xls", "xlsx"]:
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, header=0, encoding='gb18030')
        else:
            data = pd.read_excel(uploaded_file, header=0)
        st.success("数据已成功导入！")

        renderer = get_pyg_renderer(data)
        # 渲染数据探索界面,可以通过拖放构建图表。
        renderer.render_explore()
    else:
        st.warning("不支持的文件格式！")
