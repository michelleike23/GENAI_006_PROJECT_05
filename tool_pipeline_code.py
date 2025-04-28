# -*- coding: utf-8 -*-

# Load dataset
df = pd.read_csv("/content/drive/MyDrive/GenAI/labeled_contract_chunks.csv")

# Clean labels
df_clean = df.dropna(subset=["manual_label"])
df_clean = df_clean[df_clean["manual_label"].str.lower() != "none"]

# Count and plot unique labels
label_counts = df_clean["manual_label"].value_counts()

plt.figure(figsize=(10, 6))
label_counts.plot(kind="barh")
plt.title(" Distribution of Clause Labels (manual_label)")
plt.xlabel("Count")
plt.ylabel("Clause Type")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""# Creating a Legal Clause Classifier
**Exploring Logistic Regression, SVM, XGBoost and Random Forest**

SMOTE + Logistic Regression (Legal Text Classifier) - *Morgan*
"""

from google.colab import drive
drive.mount('/content/drive')



# 📚 Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Load data
file_path = "/content/drive/MyDrive/GenAI/labeled_contract_chunks.csv"
df = pd.read_csv(file_path)

# Filter out 'none' and nulls in text or labels
df_clean = df.dropna(subset=["text", "manual_label"]).copy()
df_clean = df_clean[df_clean["manual_label"].str.lower() != "none"]

#  TF-IDF vectorization on clause text
X = df_clean["text"]
y = df_clean["manual_label"]
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

#  Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train Logistic Regression
clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_resampled, y_resampled)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_micro = f1_score(y_test, y_pred, average="micro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

# 🧾 Full report
print(classification_report(y_test, y_pred))
print(f"\n Accuracy:          {accuracy:.4f}")
print(f" Macro F1-score:    {f1_macro:.4f}")
print(f" Micro F1-score:    {f1_micro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

"""SVM (Support Vector Machine)"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load labeled data
df = pd.read_csv("/content/drive/MyDrive/GenAI/labeled_contract_chunks.csv")

df_labeled = df[df["manual_label"].str.lower() != "none"].copy()
print(f" Remaining labeled samples: {len(df_labeled)}")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X = df_labeled["text"]
y = df_labeled["manual_label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Train SVM

svm = LinearSVC(
    class_weight='balanced',
    random_state=42
)

svm.fit(X_train_vec, y_train)
y_pred_svm = svm.predict(X_test_vec)

print("📊 SVM Results")
print(classification_report(y_test, y_pred_svm))
print(f"\n Accuracy:          {accuracy:.4f}")
print(f" Macro F1-score:    {f1_macro:.4f}")
print(f" Micro F1-score:    {f1_micro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

"""XGBoost"""

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score


# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),  # or len(le.classes_) if label-encoded
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
xgb.fit(X_train_vec, y_train_enc)
y_pred_xgb = xgb.predict(X_test_vec)
accuracy = accuracy_score(y_test_enc, y_pred_xgb)
f1_macro = f1_score(y_test_enc, y_pred_xgb, average="macro")
f1_micro = f1_score(y_test_enc, y_pred_xgb, average="micro")
f1_weighted = f1_score(y_test_enc, y_pred_xgb, average="weighted")


print("📊 XGBoost Results")
print(classification_report(y_test_enc, y_pred_xgb, target_names=le.classes_))
print(f"\n Accuracy:          {accuracy:.4f}")
print(f" Macro F1-score:    {f1_macro:.4f}")
print(f" Micro F1-score:    {f1_micro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

joblib.dump(xgb, "xgboost_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

"""Random Forest"""

# Imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load labeled data
df = pd.read_csv("/content/drive/MyDrive/GenAI/labeled_contract_chunks.csv")

# Clean: remove 'none' and nulls
df_clean = df.dropna(subset=["text", "manual_label"])
df_clean = df_clean[df_clean["manual_label"].str.lower() != "none"]

# Define features and labels
X = df_clean["text"]
y = df_clean["manual_label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Random Forest model
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
rf_clf.fit(X_train_vec, y_train)

# Predict
y_pred_rf = rf_clf.predict(X_test_vec)

# Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Optional: Print overall F1 scores
print(f"\n Accuracy:         {accuracy_score(y_test, y_pred_rf):.4f}")
print(f" Macro F1 Score:   {f1_score(y_test, y_pred_rf, average='macro'):.4f}")
print(f" Weighted F1 Score:{f1_score(y_test, y_pred_rf, average='weighted'):.4f}")

"""Cross Validation"""

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load & clean
df = pd.read_csv("/content/drive/MyDrive/GenAI/labeled_contract_chunks.csv")
df_clean = df.dropna(subset=["text", "manual_label"])
df_clean = df_clean[df_clean["manual_label"].str.lower() != "none"]
X = df_clean["text"]
y = df_clean["manual_label"]

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "SVM": LinearSVC(class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
}

#  Store results
results = {}

#  Loop through models
for name, model in models.items():
    print(f" Cross-validating {name}...")

    # For XGBoost, encode labels
    if name == "XGBoost":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_used = y_encoded
    else:
        y_used = y  # use raw string labels

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", model)
    ])

    # Run CV
    scores = cross_val_score(pipeline, X, y_used, cv=cv, scoring="f1_macro")
    results[name] = scores
    print(f"{name} Macro F1 (avg): {scores.mean():.4f} ± {scores.std():.4f}\n")

# Summary
results_df = pd.DataFrame(results)
print(results_df.mean().to_frame("Mean Macro F1"))

"""Model Comparison"""

import matplotlib.pyplot as plt

# F1 scores
model_names = ["Logistic Regression", "SVM", "XGBoost","Random Forest"]
mean_f1_scores = [0.6444, 0.7583, 0.7837,0.6728 ]

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, mean_f1_scores, color=["#6cace4", "#fdb913", "#00a550", "#FF6F61"])
plt.ylim(0.5, 0.85)
plt.title("Cross-Validated Mean Macro F1 Score by Model")
plt.ylabel("Macro F1 Score")

# Add text on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01,
             f"{height:.4f}", ha='center', va='bottom')

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

"""XGBoost is the best performing model

# Integrating Summarization
"""



import openai
print(openai.__version__)

# 📚 Imports
import os
import openai
import pandas as pd
import pdfplumber
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

# 🚀 Step 1: Mount Google Drive
drive.mount('/content/drive')

# 🔐 Step 2: Read API Key from file
key_path = '/content/drive/MyDrive/GenAI/openai_key.txt'
with open(key_path, 'r') as file:
    openai_api_key = file.read().strip()

openai.api_key = openai_api_key

# 🛠 Step 3: Load Trained Models
xgb_model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 📄 Step 4: Extract Full Text from Filing
def extract_text_from_pdf(pdf_path):
    """Extracts clean text from all pages in a PDF."""
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    return all_text

# 📂 Load Filing
pdf_path = "Confidentiality_Agreement.pdf"  # Adjust path if needed
full_text = extract_text_from_pdf(pdf_path)

# ✂️ Step 5: Split Full Text into Chunks
chunks = full_text.split("\n\n")

# 🧠 Step 6: Vectorize Chunks
X_chunks = vectorizer.transform(chunks)

# 🔍 Step 7: Predict Clause Types
y_pred = xgb_model.predict(X_chunks)
predicted_clauses = label_encoder.inverse_transform(y_pred)

# 🗂️ Step 8: Organize by Clause
classified_chunks = pd.DataFrame({
    "chunk": chunks,
    "predicted_clause": predicted_clauses
})

grouped = classified_chunks.groupby("predicted_clause")["chunk"].apply(lambda x: " ".join(x)).to_dict()

# ✨ Step 9: Define Summarization Function
def summarize_text(text, clause_type):
    prompt = f"""
You are an expert legal summarization assistant.
Please summarize the following text extracted from a "{clause_type}" section of a legal document.
Summarize professionally in 3-5 sentences focusing on key points.

Text:
\"\"\"
{text}
\"\"\"

Summary:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful legal summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"].strip()

# 📚 Step 10: Summarize Each Grouped Clause
structured_summaries = {}

for clause_type, text in grouped.items():
    if len(text) > 5000:
        text = text[:5000]  # Optional: truncate if too large
    summary = summarize_text(text, clause_type)
    structured_summaries[clause_type] = summary

# 📋 Step 11: Assemble Final Structured Summary
for clause, summary in structured_summaries.items():
    print(f"\n🗂️ {clause}")
    print(summary)

