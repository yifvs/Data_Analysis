import streamlit as st
from langchain.llms import OpenAI

st.title('💬 🔗 Chat with GPT-3.5')

openai_api_key = st.sidebar.text_input('OpenAI API Key', max_chars=None, key=None, type='password')

def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Ask me anything……')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key in the sidebar! You can get a key at https://platform.openai.com/account/api-keys.', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)
