import streamlit as st
from assist.chat import chat as embed_chat
from assist.bayes_chat import bayes_chat
from assist.transformer_demo import transformer_next

st.title("RepoSage Chatbot Demo")

question = st.text_input("Enter your question below:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Embedding Q&A"):
        st.write(embed_chat(question))
with col2:
    if st.button("Bayesian Q&A"):
        st.write(bayes_chat(question))
with col3:
    if st.button("Transformer Demo"):
        st.write(transformer_next(question))
