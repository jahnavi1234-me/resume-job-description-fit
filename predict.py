import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util

# --------------------
# PATHS
# --------------------
DATA_PATH = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\data\processed\test_clean.csv"
MODEL_DIR = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\models\sentence_bert"
OUTPUT_PATH = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\outputs\predictions\predictions.csv"

os.makedirs("outputs/predictions", exist_ok=True)

# --------------------
# Load data
# --------------------
df = pd.read_csv(DATA_PATH)

resume = df["resume_text"].astype(str).tolist()
jd = df["job_description_text"].astype(str).tolist()

# --------------------
# Load Sentence-BERT
# --------------------
model = SentenceTransformer(MODEL_DIR)

# --------------------
# Encode
# --------------------
resume_emb = model.encode(resume, convert_to_tensor=True)
jd_emb = model.encode(jd, convert_to_tensor=True)

# --------------------
# Cosine similarity
# --------------------
similarities = util.cos_sim(resume_emb, jd_emb).diagonal().cpu().numpy()

df["similarity_score"] = similarities

# Simple rule-based label
df["prediction"] = df["similarity_score"].apply(
    lambda x: "Good Fit" if x >= 0.6 else "No Fit"
)

# --------------------
# Save
# --------------------
df.to_csv(OUTPUT_PATH, index=False)
print("✅ Predictions saved to", OUTPUT_PATH)
