import streamlit as st

# Your existing demos
from assist.chat import chat as embed_chat
from assist.bayes_chat import bayes_chat
from assist.transformer_demo import transformer_next

# DeepSeek imports
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

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

# User input
question = st.text_input("Enter your question or prompt below:")

# Four buttons side by side, with DeepSeek first
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("DeepSeek-R1 Demo"):
        if not question.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Generating with DeepSeek…"):
                out = deepseek_gen(question, max_new_tokens=100, do_sample=True)
            st.code(out[0]["generated_text"], language="text")

with col2:
    if st.button("Embedding Q&A"):
        st.write(embed_chat(question))

with col3:
    if st.button("Bayesian Q&A"):
        st.write(bayes_chat(question))

with col4:
    if st.button("Transformer Demo"):
        st.write(transformer_next(question))

st.markdown("---")
st.caption("DeepSeek-R1, Embedding, Bayesian & Transformer demos all in one place ✅")
