import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

st.set_page_config(page_title="OpenAI GPT-3.5", layout="wide")

st.title(":blue[Chat with GPT-3.5] 🤖")

# 获取用户输入的API Key，并保存到session_state["OPENAI_API_KEY"]中
openai_api_key = st.text_input("输入你的API Key才能开始会话：", value=st.session_state["OPENAI_API_KEY"], max_chars=None, key=None, type='password')

saved = st.button("Save")

if saved:
    st.session_state["OPENAI_API_KEY"] = openai_api_key

    # 初始化ChatOpenAI对象
    chat = None
    # 如果session_state中没有定义"OPENAI_API_KEY"，则将其设置为空字符串
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = ""
    # 如果session_state中"OPENAI_API_KEY"不为空，则创建ChatOpenAI对象并传入API Key
    elif st.session_state["OPENAI_API_KEY"] != "":
        chat = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"])

    # 如果session_state中没有定义"messages"，则将其初始化为空列表
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    # 如果chat对象存在
    if chat:
        # 创建一个container，用于组织聊天界面
        with st.container():
            st.header("Ask me anything……")

            # 显示已有的聊天记录
            for message in st.session_state["messages"]:
                if isinstance(message, HumanMessage):
                    # 如果是用户发送的消息，则在聊天界面中显示为用户消息
                    with st.chat_message("user"):
                        st.markdown(message.content)
                elif isinstance(message, AIMessage):
                    # 如果是AI回复的消息，则在聊天界面中显示为AI助手消息
                    with st.chat_message("assistant"):
                        st.markdown(message.content)

            # 获取用户输入的消息，并保存到session_state["messages"]中
            prompt = st.chat_input("Prompt")
            if prompt:
                st.session_state["messages"].append(HumanMessage(content=prompt))
                with st.chat_message("user"):
                    st.markdown(prompt)
                ai_message = chat([HumanMessage(content=prompt)])
                st.session_state["messages"].append(ai_message)
                with st.chat_message("assistant"):
                    st.markdown(ai_message.content)
    else:
        with st.container():
            st.warning("Please enter your API Key to start the conversation")
