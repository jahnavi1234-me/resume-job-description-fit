from sentence_transformers import SentenceTransformer, util

MODEL_PATH = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\models\sentence_bert"
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