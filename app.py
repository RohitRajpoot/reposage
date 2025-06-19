import streamlit as st
from assist.chat import chat as chat_plugin

st.title("RepoSage Chatbot Demo")

# 1) Change the label to make it obvious we're asking a question
question = st.text_input("Ask RepoSage a question:", "")

# 2) Only run when clicked
if st.button("Ask RepoSage"):
    # 3) Pass that question into your stub
    response = chat_plugin(question)
    st.write(response)
