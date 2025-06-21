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

math_prefix = (
    "You are an expert math tutor.  Compute the derivative of f(x) = x^2·sin(x) "
    "step by step using the product rule.  Show each line of work."
)

with col1:
    if st.button("DeepSeek-R1 Math Demo"):
        if not question.strip():
            st.warning("Please enter a prompt first.")
        else:
            # 1) Build the full math prompt
            prompt = f"{math_prefix}\n\nf(x) = {question}\n\nSolution:\n"
            # 2) Call the model deterministically
            with st.spinner("Working it out…"):
                out = deepseek_gen(
                    prompt,
                    max_new_tokens=80,
                    do_sample=False,      # no random sampling
                    temperature=0.0       # fully deterministic
                )
            # 3) Display the clean, step-by-step answer
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
