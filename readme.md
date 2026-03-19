 Resume–Job Description Fit Analyzer

---

 Project Description

The Resume–Job Description Fit Analyzer is an NLP-based AI system that evaluates how well a candidate’s resume matches a given job description.
It uses both traditional machine learning and transformer-based techniques to measure semantic similarity and predict candidate-job compatibility.

This system helps automate resume screening and improves recruitment efficiency by identifying suitable candidates quickly.

---

 Problem Statement

Recruiters often spend significant time manually reviewing resumes and comparing them with job descriptions. This process is time-consuming, subjective, and inefficient.

This project aims to build an AI-powered system that automatically analyzes resumes and job descriptions to determine how well a candidate fits a specific role using NLP and transformer-based semantic similarity techniques.

---

 Features

- Semantic similarity analysis between resume and job description
- Multi-level prediction:
  -  Good Fit
  -  Potential Fit
  -  No Fit
- Transformer-based embeddings using Sentence-BERT
- Baseline model using TF-IDF + Logistic Regression
- Confusion matrix and evaluation metrics
- Interactive web application using Streamlit

---

 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Sentence-Transformers
- Transformers
- PyTorch
- Joblib
- Matplotlib
- Streamlit

---

Project Architecture
```
Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
Baseline Model (TF-IDF + Logistic Regression)
     ↓
Advanced Model (Sentence-BERT)
     ↓
Embedding Generation
     ↓
Cosine Similarity Calculation
     ↓
Prediction (Fit Category)
     ↓
Evaluation & Visualization
     ↓
Streamlit Web Application

---
```
 Folder Structure
```
resume-job-description-fit/
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│       ├── train_cleaned.csv
│       └── test_cleaned.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_logreg_model.pkl
│   └── sentence_bert/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
└── README.md

---
```
 Installation

1️ Clone Repository

git clone <your-repo-link>
cd resume-job-description-fit

2️ Install Dependencies

pip install -r requirements.txt

3️ Run the Application

streamlit run app/app.py

---

 Example Output

Input:

Resume:

«Python developer with experience in NLP and machine learning.»

Job Description:

«Looking for an NLP engineer with Python and deep learning skills.»

Output:

- Similarity Score: "0.82"
- Prediction:  Good Fit

---

 Future Improvements

- Fine-tune transformer models on domain-specific datasets
- Add keyword highlighting between resume and job description
- Deploy as API using FastAPI
- Integrate vector database (FAISS) for large-scale search
- Cloud deployment (AWS / Hugging Face Spaces)
- Build recruiter dashboard with analytics

---

 Author

Jahnavi Besabathini
Aspiring Generative AI Engineer

 
This project demonstrates how modern NLP and transformer models can automate real-world recruitment workflows, reducing manual effort and improving candidate-job matching efficiency.

---
