import streamlit as st
import requests

# Coze API 配置
API_URL = "https://api.coze.cn/open_api/v2/chat"
BEARER_TOKEN = "pat_3yBUfdwMT2OuOKjoYp7IoRHqdUIZWefU2D5DEYmJMeD9mlkMDX8S8s0vHBeg83Js"  
BOT_ID = "7385055388376907810"  

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
    response.raise_for_status()  # 如果响应状态不是 200，将抛出异常
    response_data = response.json()

    # 解析JSON数据并生成Markdown格式的字符串
    markdown_content = ""
    messages = response_data.get('messages', [])
    for message in messages:
        if message.get('role') == 'assistant' and message.get('type') == 'answer':
            # 假设answer内容是一个列表，我们将其转换为Markdown列表
            markdown_content += "### " + message.get('content', '没有找到答案。') + "\n\n"
            for item in message.get('data', []):
                if 'title' in item and 'link' in item:
                    markdown_content += "- [" + item['title'] + "](" + item['link'] + ")\n"

    return markdown_content

def main():
    st.title("📖业哥智能体 V2.0")
    st.warning("由于“业哥”的解答是基于互联网上搜索的内容，所以输出内容是JSON格式。而JSON格式本身是用于数据交换的，并不直接支持格式化显示。"
               "JSON格式的数据通常以键值对的形式存在，并且嵌套结构可以非常复杂，如果直接以文本形式展示，会显得比较混乱。"
               "后续我会研究如何解析JSON数据，然后根据数据结构生成相应的Markdown语法，让显示的内容更易读一些。", icon='⚠')
    
    # 创建文本输入框
    user_input = st.text_input("请输入你的问题：")

    # 创建按钮并获取答案
    if st.button("咨询业哥"):
        # 发送请求到 Coze API 并显示Markdown格式的答案
        markdown_answer = send_to_coze(user_input)
        st.markdown(markdown_answer, unsafe_allow_html=True)  # 使用unsafe_allow_html允许Markdown渲染

if __name__ == '__main__':
    main()
