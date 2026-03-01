import streamlit as st
from utils import predict_fit

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Resume Job Fit Analyzer",
    layout="wide"
)

st.title("🤖 Resume–Job Description Fit Analyzer")
st.write(
    "This AI model analyzes how well a resume matches a job description using Sentence-BERT embeddings."
)

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    resume_text = st.text_area(
        "📄 Paste Resume Text",
        height=300
    )

with col2:
    job_description = st.text_area(
        "💼 Paste Job Description",
        height=300
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze Fit 🚀"):

    if resume_text.strip() == "" or job_description.strip() == "":
        st.warning("Please enter both resume and job description.")
    else:
        with st.spinner("Analyzing semantic similarity..."):
            score, label = predict_fit(resume_text, job_description)

        st.success("Analysis Complete!")

        st.subheader("Result")
        st.metric("Similarity Score", f"{score:.3f}")
        st.write("Prediction:", label)