# RepoSage™ Chatbot

An MVP AI chatbot built in AML-3304 using Bayesian embeddings, a simple transformer block, and DeepSeek-R1 integration — all wired up with a GitHub-driven CI/CD pipeline to Hugging Face Spaces.

---

## 🚀 Live Demo

Try it out live:  
👉 https://huggingface.co/spaces/rohitrajpoot/reposage-chatbot

---

## 📖 Overview

**What it is:**  
- A command-line & web demo (via Streamlit) that shows:
  1. **Embedding Q&A**: nearest‐neighbor lookup in a trained token embedding (`assist/chat.py`)  
  2. **Bayesian Q&A**: frequency‐based “co-occurrence” embedding lookup (`assist/bayes_chat.py`)  
  3. **Transformer Demo**: single‐block transformer next‐token prediction (`assist/transformer_demo.py`)  
  4. **DeepSeek-R1**: calls to a 1.3B-parameter model for generative Q&A (wrapped to skip gracefully in Colab)  

**Why it matters:**  
- Demonstrates core GPT “atoms” (token → embedding → attention → generation)  
- Shows an end-to-end MLOps flow: local dev → GitHub Actions → Docker → Hugging Face Spaces

---

## ⚙️ Installation

### Local (macOS/Linux)

```bash
git clone https://github.com/rohitrajpoot/reposage.git
cd reposage

# 1) Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3) Smoke-test CLI
python -m assist.main chat "hello world"

# 4) Run Streamlit demo
streamlit run app.py
