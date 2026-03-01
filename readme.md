#  Resume–Job Description Fit Analyzer

An NLP-powered AI system that evaluates how well a candidate’s resume matches a job description using both classical machine learning and transformer-based semantic embeddings.

This project automates resume screening by measuring semantic similarity between resumes and job descriptions, helping recruiters quickly identify suitable candidates.
---

##  Project Overview

Recruiters often spend significant time manually comparing resumes with job requirements.
This project solves that problem by building an **AI-based Resume–Job Matching System** that predicts candidate-job compatibility.

The system implements:

*  Baseline NLP model (TF-IDF + Logistic Regression)
*  Advanced Transformer model (Sentence-BERT)
*  Semantic similarity scoring
*  Evaluation metrics & confusion matrix
*  Interactive Streamlit web application

---

##  Key Features

* Semantic comparison between resume and job description
* Multi-level fit prediction:

  *  Good Fit
  *  Potential Fit
  *  No Fit
* Transformer-based embeddings using Sentence-BERT
* Model comparison between classical ML and modern NLP
* Interactive AI web interface using Streamlit
* Automated evaluation and visualization

---

## Project Architecture

```
Raw Dataset
    |
Data Cleaning & Processing
     |
Baseline Model (TF-IDF + Logistic Regression)
     |
Advanced Model (Sentence-BERT Embeddings)
     |
Cosine Similarity Scoring
     |
Prediction & Evaluation
     |
Streamlit Web Application
```

---

##  Project Structure

```
resume-job-description-fit/
│
├── app/
│   ├── app.py              
│   └── utils.py            
│
├── data/
│   ├── raw/
│   └── processed/
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
```

---

##  Models Used

###  Baseline Model

**TF-IDF + Logistic Regression**

* Converts text into numerical features
* Fast and interpretable
* Used as performance baseline

---

###  Advanced Model

**Sentence-BERT (all-MiniLM-L6-v2)**

* Transformer-based semantic embeddings
* Captures contextual meaning
* Computes cosine similarity between resume and job description
* Produces improved matching performance

---

##  Evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix Visualization

Example output:

```
Accuracy: XX%
Precision: XX
Recall: XX
F1-score: XX
```

---

##  Web Application (Streamlit)

The project includes an interactive web interface where users can:

1. Paste Resume text
2. Paste Job Description
3. Click **Analyze Fit**
4. View similarity score and prediction instantly

### Run the Web App

```bash
streamlit run app/app.py
```

---

## How to Run Locally

###  Clone Repository

```bash
git clone <your-repo-link>
cd resume-job-description-fit
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Train Models (Optional)

```bash
python train.py
```

###  Launch Web App

```bash
streamlit run app/app.py
```

---

##  Dataset

Dataset sourced from HuggingFace:

**Resume–Job Description Fit Dataset**

Contains labeled pairs of:

* Resume text
* Job description text
* Fit category labels

---

## Technical Skills Demonstrated

* Natural Language Processing (NLP)
* Machine Learning
* Transformer Models
* Sentence Embeddings
* Model Evaluation
* Python Development
* Streamlit Deployment
* End-to-End ML Pipeline Design

---

## Future Improvements

* Fine-tuned domain-specific embeddings
* Resume keyword highlighting
* API deployment using FastAPI
* Cloud deployment (AWS / HuggingFace Spaces)
* Recruiter dashboard analytics

---

##  Author

**Jahnavi**

AI / NLP Enthusiast | Aspiring GenAI Engineer

---

## Why This Project Matters

This project demonstrates how modern NLP and transformer models can automate real-world recruitment workflows, reducing manual effort and improving candidate-job matching efficiency.

---
