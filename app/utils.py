import os
from sentence_transformers import SentenceTransformer, util
# Load model directly from HuggingFace Hub
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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
