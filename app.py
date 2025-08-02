# app.py

import streamlit as st
import h2o
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import yaml
from transformers import pipeline
from scipy.special import softmax

# --- 1. Load configuration ---
with open("config.yml", "r") as f:
    cfg = yaml.safe_load(f)

INDEX_MODEL    = cfg["index"]["model"]
TOP_K          = cfg["index"]["top_k"]
THRESHOLD_DEF  = cfg["qa"]["threshold"]
FALLBACK_MODEL = cfg["qa"]["fallback_model"]

# --- 2. Initialize models & data ---

@st.cache_resource
def load_h2o_classifier():
    h2o.init(max_mem_size="2G")
    # Replace with your actual model path
    return h2o.load_model("models/classifier/Actual")

@st.cache_resource
def load_sentence_embedder():
    return SentenceTransformer(INDEX_MODEL)

@st.cache_resource
def load_faiss_index():
    idx = faiss.read_index("data/deepseek.index")
    meta = pickle.load(open("data/deepseek_meta.pkl", "rb"))
    # Load a mapping from doc_id -> full text
    texts = pickle.load(open("data/deepseek_texts.pkl", "rb"))
    return idx, meta, texts

@st.cache_resource
def load_fallback_pipeline():
    # still a text2text-generation pipeline
    return pipeline("text2text-generation", model=FALLBACK_MODEL)


clf       = load_h2o_classifier()
embedder  = load_sentence_embedder()
index, meta, docs = load_faiss_index()
fallback  = load_fallback_pipeline()

# --- Helper functions ---

def classify_query(query: str):
    q = query.lower()
    # Rule-based override:
    if "install" in q or "pip install" in q:
        return "installation", 1.0

    # Otherwise use your H2O router
    emb     = embedder.encode([query]).astype("float32")
    data    = {f"emb_{i}": [emb[0][i]] for i in range(len(emb[0]))}
    hf      = h2o.H2OFrame(data)
    pred_df = clf.predict(hf).as_data_frame()
    category= pred_df["predict"][0]
    prob    = pred_df[category][0] if category in pred_df else None
    return category, prob


def retrieve_passages(query: str, category: str, top_k: int):
    # 1) encode & normalize
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    # 2) search
    D, I = index.search(q_emb, top_k)
    sims  = D[0]   # similarities
    idxs  = I[0]   # integer indices into your docs list
    # 3) pull back the actual text and file source
    passages = [docs[i] for i in idxs]
    sources  = [meta[i] for i in idxs]
    return sims, passages, sources

def bayesian_select(sims: np.ndarray):
    liks = softmax(sims)                       # Likelhoods
    prior = np.ones_like(liks) / len(liks)     # Uniform prior
    post  = prior * liks
    post /= post.sum()
    best_i = int(np.argmax(post))
    return best_i, float(post[best_i])

def answer_question(query: str, top_k: int, threshold: float):
    sims, passages, sources = retrieve_passages(query, None, top_k)
    best_i, conf = bayesian_select(sims)
    if conf >= threshold:
        return passages[best_i], sources[best_i], conf, "retrieval"
    # transformer fallback (you could pass the source for context)
    prompt = (
        f"You are an expert on RepoSage. "
        f"Answer the following question as concisely as possible:\n\n{query}"
    )
    out = fallback(
        prompt,
        max_length=256,
        do_sample=False,  # deterministic
        num_beams=4  # optional: better quality
    )
    answer_text = out[0]["generated_text"]
    return answer_text, None, None, "fallback"


# --- 4. Streamlit UI ---

st.set_page_config(page_title="RepoSage", layout="wide")
st.title("📚 RepoSage AI Assistant")

query     = st.text_input("Enter your question about the docs…")
threshold = 0.0
top_k     = st.slider("Number of passages to retrieve", 1, 10, TOP_K)

if query:
    # Classification (may include install‐override) …
    cat, prob = classify_query(query)
    st.write(f"Routed to **{cat}** (p={prob:.2f})")

    # Retrieval + Bayesian scoring …
    sims, passages, sources = retrieve_passages(query, cat, top_k)
    best_i, conf = bayesian_select(sims)

    if conf >= threshold:
        st.write(f"**Answer** (from {sources[best_i]}, conf={conf:.2f})")
        st.markdown(passages[best_i])
    else:
        # Answer-style fallback
        prompt = (
            f"You are an AI expert on RepoSage. "
            f"Please answer the following question clearly:\n\n{query}"
        )
        out = fallback(prompt, max_length=256, do_sample=False, num_beams=4)
        st.write("**Answer** via transformer fallback")
        st.markdown(out[0]["generated_text"])

