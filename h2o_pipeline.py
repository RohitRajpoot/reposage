import os
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    # 1. Start H2O cluster
    h2o.init(max_mem_size="4G")

    # 2. Load your full dataset
    pdf = pd.read_csv("data/h2o/dataset.csv")
    print(f"Loaded {len(pdf)} rows from dataset.csv")

    # 3. Compute embeddings with SentenceTransformers
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = pdf["text"].tolist()
    embs = model.encode(texts).astype("float32")
    print(f"Computed embeddings shape: {embs.shape}")

    # 4. Build a pandas DataFrame of embeddings + label
    embed_cols = [f"emb_{i}" for i in range(embs.shape[1])]
    pdf_emb = pd.DataFrame(embs, columns=embed_cols)
    pdf_emb["label"] = pdf["label"]

    # 5. Convert to H2OFrame
    hf = h2o.H2OFrame(pdf_emb)
    hf["label"] = hf["label"].asfactor()
    print("H2OFrame types:", hf.types)

    # 6. Train/Test split (80/20)
    train, test = hf.split_frame(ratios=[0.8], seed=1234)
    print(f"Train rows: {train.nrow}, Test rows: {test.nrow}")

    # 7. Run H2O AutoML on embeddings
    aml = H2OAutoML(max_models=20, seed=1, max_runtime_secs=600)
    aml.train(x=embed_cols, y="label", training_frame=train)

    # 8. Show leaderboard & save best model
    lb = aml.leaderboard
    best = aml.leader
    print("Top models:\n", lb.head(rows=10))
    os.makedirs("models", exist_ok=True)
    path = h2o.save_model(best, path="models", force=True)
    print("Saved best model to:", path)

    # 9. Evaluate on test set
    perf = best.model_performance(test)
    print("Test set performance:\n", perf)

    # 10. Shutdown H2O
    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
