# RepoSage: Hybrid AI-Centric Q\&A Pipeline

This repository contains the core components of **RepoSage**, a hybrid AI-powered study assistant that routes student queries to the right documentation sections and delivers precise answers using a blend of classification, retrieval, Bayesian scoring, and generative fallback.

---

## 🔄 Hybrid Pipeline Overview

1. **Corpus Preparation**

   * Recursively scan **all** project markdowns (`reposage/README.md`, other `*.md`).
   * Split each file into paragraphs and export to `data/classification.csv` with an empty `category` column for manual labeling.

2. **H2O Classifier (Routing Layer)**

   * After labeling paragraphs (e.g. `installation`, `usage`, `configuration`, `api_reference`, `troubleshooting`), train an H2O AutoML model to predict the category of any incoming question.
   * This routing step ensures each query is answered by the most relevant section of your docs.

3. **DeepSeek + Bayesian + Transformer Q\&A**

   * Build a FAISS index over all paragraphs (using sentence-transformer embeddings).
   * On query:

     1. **Route** via the H2O classifier to select a subset of paragraphs.
     2. **Retrieve** top‑k relevant passages with FAISS.
     3. **Rescore** with a Bayesian posterior to boost high-confidence results.
     4. **Fallback**: if the posterior confidence is below threshold, generate an answer using a compact Flan‑T5 model.

4. **Streamlit UI / Hugging Face Space**

   * The entire stack is orchestrated in `app.py` (or Gradio) as a Streamlit application.
   * Deploy on Hugging Face Spaces for instant access.
   * Users type a question, see “Routed to category: X,” and receive a clear, sourced answer.

---

## 🚀 Getting Started

### 1. Prepare the Corpus

```bash
python scripts/prepare_corpus.py \
  --input-dir reposage/ \
  --output data/classification.csv
```

Manually open `data/classification.csv` and assign each paragraph a category label.

### 2. Train the H2O Classifier

```bash
python scripts/train_classifier.py \
  --data data/classification.csv \
  --model-out models/routing_model.zip
```

### 3. Build the FAISS Index

```bash
python scripts/build_index.py \
  --data data/classification.csv \
  --index-out models/faiss_index.bin
```

### 4. Hook into Streamlit App

In `app.py`, configure paths to:

* `models/routing_model.zip`
* `models/faiss_index.bin`
* Embedding & Flan‑T5 models via your settings or environment variables.

Run locally:

```bash
streamlit run app.py
```

---

## 🔧 Directory Structure

```
├─ data/
│  └─ classification.csv       # Paragraphs + category column
├─ models/
│  ├─ routing_model.zip       # Trained H2O classifier
│  └─ faiss_index.bin         # FAISS index of embeddings
├─ scripts/
│  ├─ prepare_corpus.py       # Splits & exports paragraphs
│  ├─ train_classifier.py     # H2O AutoML training
│  └─ build_index.py          # Embedding + FAISS index builder
├─ app.py                     # Streamlit (or Gradio) Q&A app
├─ requirements.txt           # Python dependencies
└─ README.md                  # (This file)
```

---

## 🛠️ Next Steps

1. Label paragraphs in `data/classification.csv`.
2. Train or fine-tune the H2O classifier (`scripts/train_classifier.py`).
3. Integrate the routing model into your Streamlit UI so that every query is first classified, then answered from the right subset.

---

## 🤝 Contributing

1. Fork and create a feature branch.
2. Add or update scripts/tests as needed.
3. Ensure style checks (`flake8`, `black`) pass.
4. Submit a pull request detailing your enhancements.

---

## 📜 License

Licensed under MIT. See [LICENSE](LICENSE) for details.
