import streamlit as st
import requests
import json

st.set_page_config(page_title="Ask me anything（Beta）", page_icon="📖", layout="wide")

class CozeConfig:
    """集中管理Coze配置（使用Streamlit secrets）"""
    try:
        # 从secrets读取配置
        API_URL = "https://api.coze.cn/open_api/v2/chat"
        BEARER_TOKEN = st.secrets["COZE"]["BEARER_TOKEN"]  # 从secrets读取
        BOT_ID = st.secrets["COZE"]["BOT_ID"]
        DEFAULT_USER = st.secrets.get("COZE.USER", "streamlit_user")  # 带默认值
    except KeyError as e:
        st.error(f"缺少必要的配置项: {e}")
        st.stop()
    except FileNotFoundError:
        st.error("未找到secrets配置文件，请检查配置")
        st.stop()

def send_to_coze(query):
    """发送请求到Coze API并处理响应"""
    headers = {
        'Authorization': f'Bearer {CozeConfig.BEARER_TOKEN}',
        'Content-Type': 'application/json',
    }
    payload = {
        'bot_id': CozeConfig.BOT_ID,
        'user': CozeConfig.DEFAULT_USER,
        'query': query,
        'stream': False
    }

    try:
        response = requests.post(CozeConfig.API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get('messages', [])
    except requests.exceptions.RequestException as e:
        st.error(f"API请求失败: {str(e)}")
        return []

def display_chat_history():
    """展示聊天记录（优化结构化显示）"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            
            try:
                # 尝试解析JSON内容
                data = json.loads(content)
                
                # 结构化显示模板
                st.markdown(f"### {data.get('故障现象', '故障分析')}")
                
                # 分栏显示
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🛠 可能原因**\n{data.get('可能原因', '暂无')}")
                with col2:
                    st.markdown(f"**🔍 发生概率**\n{data.get('发生概率', '未知')}")
                
                # 处理步骤带序号
                steps = data.get('处理步骤', [])
                if steps:
                    st.markdown("**📝 处理步骤**")
                    for i, step in enumerate(steps, 1):
                        st.markdown(f"{i}. {step}")
                
                # 注意事项带图标
                notes = data.get('注意事项', [])
                if notes:
                    with st.expander("⚠️ 注意事项"):
                        for note in notes:
                            st.markdown(f"- {note}")
                
                # 添加浅灰色分隔线
                st.markdown("<hr style='border:1px solid #f0f0f0'>", unsafe_allow_html=True)
                
            except json.JSONDecodeError:
                # 非JSON内容原样显示
                st.write(content)

def handle_user_input():
    """处理用户输入"""
    if user_input := st.chat_input("请输入你的问题："):
        # 用户消息处理
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # 带加载状态的AI响应
        with st.spinner("正在思考中..."):
            messages = send_to_coze(user_input)

        # AI消息处理
        for message in messages:
            if message.get('role') == 'assistant':
                content = message.get('content', '没有找到答案。')
                st.session_state.messages.append({"role": "assistant", "content": content})
                st.chat_message("assistant").write(content)

def homepage():
    # 添加自定义CSS样式
    st.markdown("""
    <style>
        .stMarkdown h3 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
        }
        .stExpander .stMarkdown {
            color: #e74c3c;
        }
        .stAlert {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📖业哥智能体 V3.0")
    st.caption("🚀 基于Coze的Streamlit聊天应用")

    st.warning("内容由AI生成，仅供参考，请核实关键信息。")

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    display_chat_history()
    handle_user_input()

# 直接显示主页
homepage()
