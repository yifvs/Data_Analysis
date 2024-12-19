import streamlit as st
from openai import OpenAI

# 设置页面标题和图标
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# 初始化Session State以存储消息历史记录
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 侧边栏设置
with st.sidebar:
    model_options = ["gpt-3.5-turbo", "gpt-4"]
    selected_model = st.selectbox("Select Model", model_options)
    api_key = st.text_input(f"Enter your API KEY for {selected_model}", type="password")
    if st.button("Submit"):
        st.session_state.api_key = api_key
        st.session_state.selected_model = selected_model

def chat_with_ai(user_message):
    client = OpenAI(api_key=st.session_state.api_key)
    response = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            *st.session_state.messages,
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()

# 主页面显示聊天对话
if 'api_key' in st.session_state and 'selected_model' in st.session_state:
    st.title("Chat with AI Bot")

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**User:** {message['content']}", unsafe_allow_html=True)
        else:
            st.markdown(f"**Assistant:** {message['content']}", unsafe_allow_html=True)

    # 用户输入框
    user_input = st.text_input("Type your message here:", key="input", on_change=lambda: st.session_state.messages.append({"role": "user", "content": st.session_state.input}))
    if user_input:
        with st.spinner("Thinking..."):
            ai_response = chat_with_ai(user_input)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
    st.warning("Please select a model and enter your API KEY in the sidebar to start chatting.")




