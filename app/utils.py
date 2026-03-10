import os
from sentence_transformers import SentenceTransformer, util

# Get the base directory (one level above the current file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to your model folder inside your repo
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentence_bert")

# Load the model
model = SentenceTransformer(MODEL_PATH)
def predict_fit(resume_text, job_description):

    emb1 = model.encode([resume_text], convert_to_tensor=True)
    emb2 = model.encode([job_description], convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()

    if score >= 0.80:
        label = "✅ Good Fit"
    elif score >= 0.70:
        label = "⚠️ Potential Fit"
    else:
        label = "❌ No Fit"

    return score, label
