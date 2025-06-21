import streamlit as st
from assist.chat import chat as embed_chat
from assist.bayes_chat import bayes_chat

st.title("RepoSage Chatbot Demo")

question = st.text_input("Enter your question below:")

col1, col2 = st.columns(2)
with col1:
    if st.button("Ask Embedding RepoSage"):
        answer = embed_chat(question)
        st.write(answer)

with col2:
    if st.button("Ask Bayesian RepoSage"):
        answer = bayes_chat(question)
        st.write(answer)
