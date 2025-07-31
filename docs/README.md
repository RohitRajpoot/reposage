# RepoSage – AML3304 AI Product Engineering

## Overview
RepoSage is an AI-centric study assistant that uses semantic search (DeepSeek), a Bayesian Q&A layer, and a transformer fallback to answer student queries with traceability and transparency.

## Features
- **DeepSeek Index** of lecture slides, code, and readings  
- **Bayesian Posterior Scoring** for precise answer selection  
- **Transformer Fallback** via HF text2text when confidence is low  
- **CLI & REST API** for local and web deployment  
- **CI/CD** via GitHub Actions → Hugging Face Spaces

## Getting Started

### Prerequisites
- Python 3.8+  
- `faiss`, `sentence-transformers`, `transformers`, `openai-whisper`

### Installation
```bash
git clone https://github.com/RohitRajpoot/reposage.git
cd reposage
pip install -r requirements.txt

### Usage
# Build the index
reposage index

# Query
reposage query "What is DeepSeek?" --threshold 0.3

