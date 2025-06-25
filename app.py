import streamlit as st

# Your existing demos
from assist.chat import chat as embed_chat
from assist.bayes_chat import bayes_chat
from assist.transformer_demo import transformer_next

# DeepSeek imports
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
# Retrieval imports
from sentence_transformers import SentenceTransformer
import torch

st.set_page_config(page_title="RepoSage All-in-One Demo", layout="wide")
st.title("🤖 RepoSage Unified Demo")

# Cache and load DeepSeek-R1
@st.cache_resource
def load_deepseek():
    model_name = "deepseek-ai/DeepSeek-Coder-1.3B-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    return TextGenerationPipeline(model=model, tokenizer=tokenizer)

deepseek_gen = load_deepseek()

# Cache and load training corpus passages
@st.cache_data
def load_passages(path="RepoSage Training.txt"):
    text = open(path, encoding="utf8").read()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras

# Cache and embed passages
@st.cache_resource
def embed_passages(passages):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(passages, convert_to_tensor=True)
    return encoder, passages, embeddings

# Prepare RAG resources
_passages = load_passages()
_encoder, passages, passage_embs = embed_passages(_passages)

# User input
title = st.text_input("Enter your question or prompt below:")

# Define columns for five demos
col1, col2, col3, col4, col5 = st.columns(5)

# Math demo in col1
with col1:
    if st.button("DeepSeek-R1 Math Demo"):
        if not title.strip():
            st.warning("Please enter a prompt first.")
        else:
            prompt = f"You are an expert math tutor. Compute the derivative of f(x) = {title} step by step using the product rule. Solution:\n"
            with st.spinner("Working it out…"):
                out = deepseek_gen(prompt, max_new_tokens=80, do_sample=False, temperature=0.0)
            st.code(out[0]["generated_text"], language="text")

# RAG-augmented demo in col2
with col2:
    if st.button("DeepSeek-R1 RAG Demo"):
        if not title.strip():
            st.warning("Please enter a question first.")
        else:
            q_emb = _encoder.encode(title, convert_to_tensor=True)
            sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), passage_embs)
            topk = torch.topk(sims, k=min(3, len(passages))).indices.tolist()
            context = "\n\n".join(passages[i] for i in topk)
            prompt = f"Use these notes to answer the question:\n\n{context}\n\nQ: {title}\nA:"
            with st.spinner("Retrieving & generating…"):
                out = deepseek_gen(prompt, max_new_tokens=100, do_sample=False)
            st.write(out[0]["generated_text"])

# Embedding Q&A in col3
with col3:
    if st.button("Embedding Q&A"):
        st.write(embed_chat(title))

# Bayesian Q&A in col4
with col4:
    if st.button("Bayesian Q&A"):
        st.write(bayes_chat(title))

# Transformer Demo in col5
with col5:
    if st.button("Transformer Demo"):
        st.write(transformer_next(title))

st.markdown("---")
st.caption("DeepSeek-R1 Math, RAG, Embedding, Bayesian & Transformer demos all in one place ✅")
