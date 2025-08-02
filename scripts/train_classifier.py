import os
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sentence_transformers import SentenceTransformer

# 1. Init H2O
h2o.init(max_mem_size="4G")

# 2. Load and embed the text
df = pd.read_csv("data/classification_balanced.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")
embs  = model.encode(df["text"].tolist()).astype("float32")

# 3. Build a DataFrame of embeddings + category
cols = [f"emb_{i}" for i in range(embs.shape[1])]
df_emb = pd.DataFrame(embs, columns=cols)
df_emb["category"] = df["category"]

# 4. Convert to H2OFrame & set target
hf = h2o.H2OFrame(df_emb)
hf["category"] = hf["category"].asfactor()

# 5. Train/Test split
train, test = hf.split_frame(ratios=[0.8], seed=1234)

# 6. AutoML classification
aml = H2OAutoML(max_models=10, seed=1, max_runtime_secs=300, balance_classes=True)
aml.train(x=cols, y="category", training_frame=train)

# 7. Save the best model
os.makedirs("models", exist_ok=True)
best = aml.leader
h2o.save_model(best, path="models/classifier", force=True)

print("Classifier leaderboard:\n", aml.leaderboard)
h2o.cluster().shutdown()
