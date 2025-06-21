import streamlit as st
from assist.chat import chat as chat_plugin

st.title("RepoSage Chatbot Demo")

# 1) Change the label to make it obvious we're asking a question
question = st.text_input("Ask RepoSage a question:", "")

col1, col2 = st.columns(2)
with col1:
    if st.button("Ask Embedding RepoSage"):
        st.write(embed_chat(question))

with col2:
    if st.button("Ask Bayesian RepoSage"):
        st.write(bayes_chat(question))