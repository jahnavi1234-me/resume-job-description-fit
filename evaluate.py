import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------
# PATHS
# --------------------
DATA_PATH = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\data\processed\test_clean.csv"
MODEL_DIR = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\models"
FIG_PATH = r"C:\Users\DELL\OneDrive\Desktop\resume-job-description-fit\outputs\figures\confusion_matrix.png"

# --------------------
# Load data
# --------------------
df = pd.read_csv(DATA_PATH)

X_text = df["resume_text"] + " " + df["job_description_text"]
y_true = df["label"]

# --------------------
# Load TF-IDF model
# --------------------
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "tfidf_logreg_model.pkl"))

X_vec = vectorizer.transform(X_text)
y_pred = model.predict(X_vec)

# --------------------
# Confusion Matrix
# --------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("TF-IDF Confusion Matrix")
plt.savefig(FIG_PATH)
plt.close()

print("✅ Confusion matrix saved at:", FIG_PATH)
