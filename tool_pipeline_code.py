# -*- coding: utf-8 -*-

# 📚 Imports
import os
import openai
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import pdfplumber
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 🛠 Load Dataset
df = pd.read_csv("labeled_contract_chunks.csv")  # Make sure the CSV is in the same repo folder

# 🧼 Clean Labels
df_clean = df.dropna(subset=["manual_label", "text"])
df_clean = df_clean[df_clean["manual_label"].str.lower() != "none"]

# 📊 Plot Label Distribution
label_counts = df_clean["manual_label"].value_counts()
plt.figure(figsize=(10, 6))
label_counts.plot(kind="barh", color="#6cace4")
plt.title("Distribution of Clause Labels (manual_label)")
plt.xlabel("Count")
plt.ylabel("Clause Type")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ✨ TF-IDF Vectorization
X = df_clean["text"]
y = df_clean["manual_label"]
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# 🔀 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# ⚖️ Balance Classes with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Logistic Regression Model
lr_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
lr_clf.fit(X_resampled, y_resampled)
y_pred_lr = lr_clf.predict(X_test)

print("📊 Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))

# SVM Model
svm = LinearSVC(class_weight='balanced', random_state=42)
svm.fit(X_resampled, y_resampled)
y_pred_svm = svm.predict(X_test)

print("📊 SVM Results")
print(classification_report(y_test, y_pred_svm))

# XGBoost Model
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_train, y_train_enc)
y_pred_xgb = xgb.predict(X_test)
print("📊 XGBoost Results")
print(classification_report(y_test_enc, y_pred_xgb, target_names=le.classes_))

# Random Forest Model
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("🌲 Random Forest Results")
print(classification_report(y_test, y_pred_rf))

# 🧠 Save the Trained Models
joblib.dump(xgb, "xgboost_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

# 📈 Cross-Validation on Models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "SVM": LinearSVC(class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"Cross-validating {name}...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", model)
    ])
    if name == "XGBoost":
        scores = cross_val_score(pipeline, X, le.fit_transform(y), cv=cv, scoring="f1_macro")
    else:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    results[name] = scores
    print(f"{name} Macro F1 (avg): {scores.mean():.4f} ± {scores.std():.4f}\n")

results_df = pd.DataFrame(results)
print(results_df.mean().to_frame("Mean Macro F1"))

# 📊 Model Comparison
model_names = ["Logistic Regression", "SVM", "XGBoost", "Random Forest"]
mean_f1_scores = results_df.mean()

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, mean_f1_scores, color=["#6cace4", "#fdb913", "#00a550", "#FF6F61"])
plt.ylim(0.5, 0.85)
plt.title("Cross-Validated Mean Macro F1 Score by Model")
plt.ylabel("Macro F1 Score")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f"{height:.4f}", ha='center', va='bottom')
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 📂 Load Filing
def extract_text_from_pdf(pdf_path):
    """Extracts clean text from a PDF."""
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    return all_text

# 🔥 Summarization Section
openai.api_key = os.getenv("OPENAI_API_KEY")  # You should set it in your Streamlit Secrets or .env

# Load trained components
xgb_model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Example PDF file
pdf_path = "Confidentiality_Agreement.pdf"
full_text = extract_text_from_pdf(pdf_path)
chunks = full_text.split("\n\n")

X_chunks = vectorizer.transform(chunks)
y_pred = xgb_model.predict(X_chunks)
predicted_clauses = label_encoder.inverse_transform(y_pred)

classified_chunks = pd.DataFrame({
    "chunk": chunks,
    "predicted_clause": predicted_clauses
})

grouped = classified_chunks.groupby("predicted_clause")["chunk"].apply(lambda x: " ".join(x)).to_dict()

# Summarize using GPT-4
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

structured_summaries = {}
for clause_type, text in grouped.items():
    if len(text) > 5000:
        text = text[:5000]
    summary = summarize_text(text, clause_type)
    structured_summaries[clause_type] = summary

# Final output
for clause, summary in structured_summaries.items():
    print(f"\n🗂️ {clause}")
    print(summary)

