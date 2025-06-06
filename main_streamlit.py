import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import requests
import json
from typing import Dict, Any, Optional
# 设置页面配置
st.set_page_config(layout="wide", page_title="Data Analysis", page_icon="📊")

# LangChain相关导入
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("⚠️ LangChain未安装，请运行: pip install langchain")


# 定义系统提示模板
SYSTEM_TEMPLATE = """
你是一个专业的数据分析专家。你的任务是：
- 理解用户关于数据的问题
- 执行适当的数据分析
- 提供清晰的数据解释和洞察
- 回答数据相关的统计问题

可用的数据列：{columns}
数据概览：
- 行数：{rows}
- 列数：{cols}
- 数据样本：
{sample}

请用中文回答，保持专业、简洁和有洞察力。专注于数据分析。
"""

def setup_deepseek_llm(api_key, model_name):
    """设置DeepSeek LLM"""
    if not LANGCHAIN_AVAILABLE:
        return None
        
    try:
        # 创建自定义LangChain集成
        from langchain.chat_models.base import BaseChatModel
        from langchain.schema.messages import BaseMessage
        from langchain.schema.output import ChatGeneration, ChatResult
        from typing import List, Optional, Dict, Any
        
        class DeepSeekChatModel(BaseChatModel):
            api_key: str
            model_name: str
            temperature: float = 0.7
            
            def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager = None, **kwargs) -> ChatResult:
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # 转换LangChain消息格式为DeepSeek格式
                deepseek_messages = []
                for message in messages:
                    role = message.type
                    if role == "human":
                        role = "user"
                    elif role == "ai":
                        role = "assistant"
                    # 'system' type is already correct
                    deepseek_messages.append({
                        "role": role,
                        "content": message.content
                    })
                
                data = {
                    "model": self.model_name,
                    "messages": deepseek_messages,
                    "temperature": self.temperature
                }
                
                if stop:
                    data["stop"] = stop
                
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                
                response_data = response.json()
                message_content = response_data["choices"][0]["message"]["content"]
                
                generation = ChatGeneration(
                    message=SystemMessage(content=message_content),
                    generation_info={"finish_reason": response_data["choices"][0].get("finish_reason")}
                )
                
                return ChatResult(generations=[generation])
            
            @property
            def _llm_type(self) -> str:
                return "deepseek-chat"
        
        return DeepSeekChatModel(api_key=api_key, model_name=model_name)
    except Exception as e:
        st.error(f"设置DeepSeek LLM失败: {str(e)}")
        return None

def call_llm_with_data(llm, user_input, data, function_call=None):
    """使用LLM处理数据分析请求"""
    if llm is None:
        return "LLM未正确配置，请检查API密钥和模型设置。"
    
    try:
        # 准备数据概览
        columns_list = data.columns.tolist()
        sample_data = data.head(3).to_string(max_cols=5, max_colwidth=15)
        
        # 创建提示模板
        system_template = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
        human_template = HumanMessagePromptTemplate.from_template("{question}")
        chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
        
        # 格式化提示
        messages = chat_prompt.format_messages(
            columns=", ".join(columns_list),
            rows=len(data),
            cols=len(columns_list),
            sample=sample_data,
            question=user_input
        )
        
        # 调用LLM
        response = llm.invoke(messages) # 更新调用方式
        return response.content
    except Exception as e:
        return f"LLM调用出错: {str(e)}"

def call_deepseek_api(prompt, model, api_key):
    """调用DeepSeek API"""
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个专业的数据分析助手，请用中文回答用户关于数据的问题。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
        
    except Exception as e:
        return f"API调用失败: {str(e)}"

def process_chat_input(user_input, data, model_provider="LangChain", deepseek_model=None, deepseek_api_key=None):
    """处理用户聊天输入 - 基于LangChain实现"""
    response = {'role': 'assistant', 'content': ''}
    
    try:
        # 使用LangChain + DeepSeek模式
        if model_provider == "LangChain" and LANGCHAIN_AVAILABLE:
            # 必须使用用户提供的API Key
            api_key = deepseek_api_key
            if not api_key:
                response['content'] = "❌ 请在侧边栏输入DeepSeek API密钥后再使用AI分析功能。"
                return response
            model = deepseek_model or "deepseek-chat"
            
            # 设置LLM
            llm = setup_deepseek_llm(api_key, model)
            if llm is None:
                response['content'] = "LLM设置失败，请检查配置。"
                return response
            
            # 获取LLM分析结果（仅数据分析）
            analysis_result = call_llm_with_data(llm, user_input, data)
            response['content'] = analysis_result
        
        # 备选：DeepSeek API模式
        elif model_provider == "DeepSeek" and deepseek_api_key:
            # 构建包含数据信息的提示（限制数据量避免token超限）
            columns_info = ', '.join(data.columns.tolist()[:10])  # 只显示前10列
            if len(data.columns) > 10:
                columns_info += f"... (共{len(data.columns)}列)"
            
            # 只显示前3行数据的简化版本
            sample_data = data.head(3).to_string(max_cols=5, max_colwidth=20)
            
            data_info = f"""数据概览：
            - 行数：{len(data)}
            - 列数：{len(data.columns)}
            - 主要列名：{columns_info}
            - 样本数据（前3行）：\n{sample_data}
            
            用户问题：{user_input}
            
            请基于以上数据概览回答用户问题。如需更详细信息，请告知。"""
            
            api_response = call_deepseek_api(data_info, deepseek_model, deepseek_api_key)
            response['content'] = api_response

            pass

        
        # 默认模式：基本数据查询
        else:
            if "多少行" in user_input or "行数" in user_input:
                response['content'] = f"数据共有 {len(data)} 行，{len(data.columns)} 列。"
            elif "多少列" in user_input or "列数" in user_input:
                response['content'] = f"数据共有 {len(data.columns)} 列，列名为：{', '.join(data.columns.tolist())}"
            elif "前" in user_input and "行" in user_input:
                try:
                    num = int(''.join(filter(str.isdigit, user_input)))
                    if num > 0:
                        response['content'] = f"前{num}行数据：\n\n{data.head(num).to_string()}"
                    else:
                        response['content'] = f"前5行数据：\n\n{data.head().to_string()}"
                except:
                    response['content'] = f"前5行数据：\n\n{data.head().to_string()}"
            elif "统计" in user_input or "描述" in user_input:
                response['content'] = f"数据统计信息：\n\n{data.describe().to_string()}"
            elif "缺失" in user_input or "空值" in user_input:
                missing_info = data.isnull().sum()
                response['content'] = f"缺失值统计：\n\n{missing_info.to_string()}"
            else:
                response['content'] = "建议安装LangChain以获得更好的数据分析体验。当前仅支持基本数据查询功能。"
                
    except Exception as e:
        response['content'] = f"处理出错：{str(e)}"
    
    return response

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
    
    # 使用Plotly的默认颜色序列，提供更专业的颜色搭配
    # Use Plotly's default color sequence for more professional color schemes
    # colors = px.colors.qualitative.Plotly  # 或者可以选择其他颜色序列如：Set1, Set2, Pastel1, Dark2等
    colors = px.colors.qualitative.Set1
    
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
        
        # 文件读取函数
        def smart_file_reader(file, file_ext, header_row):
            """
            智能文件读取函数，提供多种索引列处理方案
            
            Args:
                file: 上传的文件对象
                file_ext: 文件扩展名
                header_row: 表头行数
            
            Returns:
                pandas.DataFrame: 处理后的数据框
            """
            try:
                # 首先尝试不指定索引列读取文件，获取列信息
                if file_ext == "csv":
                    temp_data = pd.read_csv(file, header=int(header_row), dtype='str', encoding='gb18030')
                else:  # xlsx
                    temp_data = pd.read_excel(file, header=int(header_row), dtype='str')
                
                # 检查是否存在常见的时间列名
                time_columns = ['Time', 'TIME', 'time', 'Timestamp', 'TIMESTAMP', 'timestamp', 
                               'DateTime', 'DATETIME', 'datetime', '时间', '时刻']
                
                found_time_column = None
                for col in time_columns:
                    if col in temp_data.columns:
                        found_time_column = col
                        break
                
                if found_time_column:
                    # 找到时间列，使用它作为索引
                    st.info(f"✅ 自动检测到时间列：{found_time_column}，将其设为索引列")
                    if file_ext == "csv":
                        data = pd.read_csv(file, index_col=found_time_column, header=int(header_row), dtype='str', encoding='gb18030')
                    else:
                        data = pd.read_excel(file, index_col=found_time_column, header=int(header_row), dtype='str')
                    return data
                else:
                    # 未找到时间列，提供用户选择
                    st.warning("⚠️ 未检测到标准时间列名，请选择索引处理方式：")
                    
                    index_option = st.radio(
                        "索引列处理方式：",
                        options=[
                            "使用默认数字索引（0, 1, 2, ...）",
                            "手动指定索引列",
                            "使用第一列作为索引"
                        ],
                        key="index_option"
                    )
                    
                    if index_option == "使用默认数字索引（0, 1, 2, ...）":
                        st.success("✅ 使用默认数字索引")
                        return temp_data
                    
                    elif index_option == "手动指定索引列":
                        selected_index_col = st.selectbox(
                            "请选择要作为索引的列：",
                            options=temp_data.columns.tolist(),
                            key="manual_index_col"
                        )
                        
                        if st.button("确认使用选定的索引列", key="confirm_index"):
                            st.success(f"✅ 使用 {selected_index_col} 作为索引列")
                            if file_ext == "csv":
                                data = pd.read_csv(file, index_col=selected_index_col, header=int(header_row), dtype='str', encoding='gb18030')
                            else:
                                data = pd.read_excel(file, index_col=selected_index_col, header=int(header_row), dtype='str')
                            return data
                        else:
                            st.info("👆 请点击确认按钮以应用索引列设置")
                            return None
                    
                    elif index_option == "使用第一列作为索引":
                        first_col = temp_data.columns[0]
                        st.success(f"✅ 使用第一列 '{first_col}' 作为索引")
                        if file_ext == "csv":
                            data = pd.read_csv(file, index_col=0, header=int(header_row), dtype='str', encoding='gb18030')
                        else:
                            data = pd.read_excel(file, index_col=0, header=int(header_row), dtype='str')
                        return data
                    
                    return None
                    
            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")
                st.info("💡 建议检查文件格式、编码或表头设置")
                return None
        
        if file_extension in ["csv", "xlsx"]:
            data = smart_file_reader(uploaded_file, file_extension, header)
            
            if data is not None:
                st.success("🎉 数据已成功导入！")
            else:
                return  # 如果数据读取失败或用户未完成配置，直接返回
        else:
            st.sidebar.warning("❌ 不支持的文件格式！请上传CSV或Excel文件")
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
            st.markdown("### 📊 数据分析配置")
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
            
            st.markdown("---")
            st.markdown("### 🤖 Chat with Excel")
            
            # 启用Chat功能
            enable_chat = st.checkbox(':blue[启用智能对话分析]', value=False)
            
            if enable_chat:
                # 大模型选择
                model_provider = st.selectbox(
                    "选择大模型提供商：",
                    options=["LangChain", "DeepSeek"],
                    help="选择您偏好的大模型接口方式"
                )
                
                # DeepSeek模型选择（两种模式都需要）
                deepseek_model = st.selectbox(
                    "选择DeepSeek模型：",
                    options=["deepseek-chat", "deepseek-reasoner"],
                    format_func=lambda x: "DeepSeek V3" if x == "deepseek-chat" else "DeepSeek R1"
                )
                
                # API Key输入（两种模式都需要）
                deepseek_api_key = st.text_input(
                    "DeepSeek API Key：",
                    type="password",
                    help="请输入您的DeepSeek API密钥"
                )
                
                if deepseek_api_key:
                    st.success("✅ DeepSeek API配置完成")
                else:
                    st.warning("⚠️ 请输入DeepSeek API Key")
                    
                # 显示选择的模式信息
                if model_provider == "LangChain":
                    st.info("🔗 使用LangChain框架调用DeepSeek API")
                else:
                    st.info("🚀 直接调用DeepSeek API")

            else:
                model_provider = None
                deepseek_model = None
                deepseek_api_key = None
            
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
                        fig.update_xaxes(row=row_num, col=col_num) 
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
                        fig.update_xaxes(row=row_num, col=col_num) 
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
                    fig.update_xaxes(rangeslider=dict(visible=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black', tickmode='linear', dtick=300),
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
                        data[column] = data[column].interpolate(method='linear')
                    
                    # 创建支持三轴的图表
                    fig = go.Figure()
                    
                    # 轴选择界面
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        secondary_axis = st.selectbox(":blue[请选择作为副轴的列（如果有的话）]", options=[None] + numeric_columns)
                    with col2:
                        third_axis = st.selectbox(":green[请选择作为第三轴的列（如果有的话）]", options=[None] + numeric_columns)
                    with col3:
                        if secondary_axis and third_axis and secondary_axis == third_axis:
                            st.error("副轴和第三轴不能选择相同的列！")
                            third_axis = None
                    
                    # 计算主轴列
                    excluded_columns = [col for col in [secondary_axis, third_axis] if col is not None]
                    primary_axis_columns = [col for col in numeric_columns if col not in excluded_columns]
                    
                    # 定义轴的颜色
                    axis_colors = {
                        'primary': ['blue', 'navy', 'darkblue', 'steelblue'],
                        'secondary': ['red', 'crimson', 'darkred', 'indianred'],
                        'third': ['green', 'darkgreen', 'forestgreen', 'seagreen']
                    }
                    
                    # 添加主轴数据
                    for i, column in enumerate(primary_axis_columns):
                        color = axis_colors['primary'][i % len(axis_colors['primary'])]
                        fig.add_trace(go.Scatter(
                            x=data.index, 
                            y=data[column], 
                            mode='lines', 
                            name=f"{column} (主轴)", 
                            line=dict(width=2, color=color),
                            yaxis='y'
                        ))
                    
                    # 添加副轴数据
                    if secondary_axis:
                        color = axis_colors['secondary'][0]
                        fig.add_trace(go.Scatter(
                            x=data.index, 
                            y=data[secondary_axis], 
                            mode='lines', 
                            name=f"{secondary_axis} (副轴)", 
                            line=dict(width=2, color=color),
                            yaxis='y2'
                        ))
                    
                    # 添加第三轴数据
                    if third_axis:
                        color = axis_colors['third'][0]
                        fig.add_trace(go.Scatter(
                            x=data.index, 
                            y=data[third_axis], 
                            mode='lines', 
                            name=f"{third_axis} (第三轴)", 
                            line=dict(width=2, color=color),
                            yaxis='y3'
                        ))
                    
                    # 设置悬停标签颜色
                    for i, trace in enumerate(fig.data):
                        if '主轴' in trace.name:
                            trace.hoverlabel = dict(bgcolor='lightblue', font=dict(size=14, color='black', family='Arial'))
                        elif '副轴' in trace.name:
                            trace.hoverlabel = dict(bgcolor='lightcoral', font=dict(size=14, color='black', family='Arial'))
                        elif '第三轴' in trace.name:
                            trace.hoverlabel = dict(bgcolor='lightgreen', font=dict(size=14, color='black', family='Arial'))
                    
                    # 更新布局
                    layout_update = {
                        'showlegend': True, 
                        'width': 1200, 
                        'height': 600, 
                        'hovermode': 'x',
                        'xaxis': dict(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor='lightgray', 
                            griddash='dot', 
                            showline=True, 
                            linewidth=1, 
                            linecolor='black', 
                            tickmode='linear', 
                            dtick=300,
                            tickangle=45,
                            rangeslider=dict(visible=True, thickness=0.1)
                        ),
                        'yaxis': dict(
                             title=dict(text='主轴', font=dict(color='blue')),
                             tickfont=dict(color='blue'),
                             showgrid=True, 
                             gridwidth=1, 
                             gridcolor='lightgray', 
                             griddash='dot', 
                             showline=True, 
                             linewidth=1, 
                             linecolor='blue'
                         )
                    }
                    
                    # 添加副轴配置
                    if secondary_axis:
                        layout_update['yaxis2'] = dict(
                             title=dict(text='副轴', font=dict(color='red')),
                             tickfont=dict(color='red'),
                             showgrid=False,
                             showline=True, 
                             linewidth=1, 
                             linecolor='red',
                             overlaying='y', 
                             side='right'
                         )
                    
                    # 添加第三轴配置
                    if third_axis:
                        layout_update.update({
                            'xaxis': dict(
                                domain=[0, 0.9]  # 压缩主图区域到90%
                            ),
                            'yaxis3': dict(
                                title=dict(text='第三轴', font=dict(color='green')),
                                tickfont=dict(color='green'),
                                showgrid=False,
                                showline=True,
                                linewidth=1,
                                linecolor='green',
                                overlaying='y',
                                side='right',
                                anchor='free',
                                position=0.95  # 在压缩后的区域内，使视觉上第三轴在图表外侧
                            ),
                            'margin': dict(r=100)
                        })
                    
                    fig.update_layout(**layout_update)
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

        # Chat with Excel 功能实现
        if 'enable_chat' in locals() and enable_chat and data is not None:
            st.markdown("---")
            st.markdown("### 🤖 Chat with Excel - 智能数据对话")
            
            # 初始化聊天历史
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # 显示聊天历史
            chat_container = st.container()
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    if message['role'] == 'user':
                        st.markdown(f"**🙋 用户:** {message['content']}")
                    else:
                        st.markdown(f"**🤖 助手:** {message['content']}")
            
            # 聊天输入
            user_input = st.text_input(
                "💬 请输入您的问题：",
                key="chat_input"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                send_button = st.button("发送", type="primary")
            with col2:
                clear_button = st.button("清空对话")
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
            
            if send_button and user_input:
                # 添加用户消息到历史
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # 处理用户输入 - 使用侧边栏配置的参数
                response = process_chat_input(
                    user_input, 
                    data, 
                    model_provider=model_provider,
                    deepseek_model=deepseek_model,
                    deepseek_api_key=deepseek_api_key
                )
                
                # 添加助手回复到历史
                st.session_state.chat_history.append(response)
                
                st.rerun()
                        
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 版权声明        
    Copyright © 2024 海航航空技术有限公司. All Rights Reserved.                          
    本应用程序受著作权法和其他知识产权法保护。未经授权，禁止复制、修改或分发本程序的任何部分。
    """)
    st.sidebar.markdown("Report Bug : kangy_wang@hnair.com")
    # 添加一些空行来确保版权信息在底部
    st.sidebar.markdown("<br>" * 5, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
