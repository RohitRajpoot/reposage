import streamlit as st
from pathlib import Path
from assist.chat import chat as chat_plugin

st.title("RepoSage Chatbot Demo")

repo_input = st.text_input("Path to your repo:", ".")
if st.button("Ask RepoSage"):
    result = chat_plugin(Path(repo_input))
    st.write(result)
