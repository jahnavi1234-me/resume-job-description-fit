 train.py # ===============================
# Sentence-BERT Training Notebook
# ===============================

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Data
# -------------------------------
DATA_PATH = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\data\processed\train_clean.csv"

df = pd.read_csv(DATA_PATH)

# Safety check
df = df.dropna(subset=["resume_text", "job_description_text", "label"])

# Label mapping
label_map = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2
}
df["label"] = df["label"].map(label_map)

# -------------------------------
# Train / Validation split
# -------------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# -------------------------------
# Prepare SBERT Training Examples
# -------------------------------
train_examples = [
    InputExample(
        texts=[row["resume_text"], row["job_description_text"]],
        label=float(row["label"])
    )
    for _, row in train_df.iterrows()
]

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16
)

# -------------------------------
# Load Sentence-BERT Model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Loss (Cosine similarity)
train_loss = losses.CosineSimilarityLoss(model)

# -------------------------------
# Train Model (FAST)
# -------------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,                 # keep small (CPU-friendly)
    warmup_steps=100,
    show_progress_bar=True
)

# -------------------------------
# Evaluation
# -------------------------------
def predict_similarity(resumes, jobs):
    emb1 = model.encode(resumes, convert_to_tensor=True)
    emb2 = model.encode(jobs, convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(emb1, emb2)
    return cosine_scores.cpu().numpy()

scores = predict_similarity(
    val_df["resume_text"].tolist(),
    val_df["job_description_text"].tolist()
)

# Convert similarity → classes
def score_to_label(score):
    if score < 0.4:
        return 0
    elif score < 0.7:
        return 1
    else:
        return 2

preds = [score_to_label(s) for s in scores]

print(classification_report(val_df["label"], preds))

# -------------------------------
# Save Model
# -------------------------------
MODEL_DIR = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\models\sentence_bert"
model.save(MODEL_DIR)

print("✅ Sentence-BERT model trained and saved")
