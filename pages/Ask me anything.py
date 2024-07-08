import streamlit as st
import requests

st.set_page_config(page_title="Ask me anything", page_icon="📖", layout="wide")

# Coze API configuration
API_URL = "https://api.coze.cn/open_api/v2/chat"
BEARER_TOKEN = "pat_3yBUfdwMT2OuOKjoYp7IoRHqdUIZWefU2D5DEYmJMeD9mlkMDX8S8s0vHBeg83Js"  # Replace with your Bearer Token
BOT_ID = "7385055388376907810"  # Replace with your Bot ID

def send_to_coze(query):
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'Content-Type': 'application/json',
    }
    data = {
        'bot_id': BOT_ID,
        'user': 'user_example',
        'query': query,
        'stream': False
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    response_data = response.json()

    messages = response_data.get('messages', [])
    return messages

st.title("📖业哥智能体 V2.0")
st.caption("🚀 A Streamlit chatbot powered by Coze")
st.warning("由于“业哥”的解答是基于互联网上搜索的内容，所以输出内容是JSON格式。而JSON格式本身是用于数据交换的，并不直接支持格式化显示。"
           "JSON格式的数据通常以键值对的形式存在，并且嵌套结构可以非常复杂，如果直接以文本形式展示，会显得比较混乱。"
           "后续我会研究如何解析JSON数据，然后根据数据结构生成相应的Markdown语法，让显示的内容更易读一些。", icon='⚠')

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("请输入你的问题："):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    messages = send_to_coze(user_input)
    for message in messages:
        if message.get('role') == 'assistant':
            content = message.get('content', '没有找到答案。')
            st.session_state.messages.append({"role": "assistant", "content": content})
            st.chat_message("assistant").write(content)
