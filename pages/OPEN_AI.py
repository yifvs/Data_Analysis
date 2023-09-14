import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from cachetools import cached, TTLCache

# 初始化ChatOpenAI对象
@cached(cache=TTLCache(maxsize=1, ttl=3600))
def initialize_chat():
    if "OPENAI_API_KEY" in st.session_state and st.session_state["OPENAI_API_KEY"]:
        return ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"])
    return None

def is_api_key_saved():
    return "OPENAI_API_KEY" in st.session_state and st.session_state["OPENAI_API_KEY"]

# 导航栏
st.sidebar.title("Chat with GPT-3.5 🤖")

# 获取用户输入的API Key，并保存到session_state["OPENAI_API_KEY"]中
openai_api_key = st.sidebar.text_input("Enter your API Key to start the conversation:", value=st.session_state.get("OPENAI_API_KEY", ""), max_chars=None, key=None, type='password')
saved = st.sidebar.button("Save")
if saved:
    st.session_state["OPENAI_API_KEY"] = openai_api_key

# 如果chat对象存在
chat = initialize_chat()
if chat:
    # 显示聊天界面
    st.header("Ask me anything……")

    # 显示已有的聊天记录
    for message in st.session_state.get("messages", []):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # 获取用户输入的消息，并保存到session_state["messages"]中
    prompt = st.chat_input("Prompt")
    if prompt:
        st.session_state.setdefault("messages", []).append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)
        ai_message = chat([HumanMessage(content=prompt)])
        st.session_state["messages"].append(ai_message)
        with st.chat_message("assistant"):
            st.write(ai_message.content)
else:
    st.warning("Please enter your API Key to start the conversation")
