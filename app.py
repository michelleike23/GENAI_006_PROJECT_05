# -*- coding: utf-8 -*-

# 📚 Imports
import os
import torch
import pandas as pd
import joblib
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

# === 🛠 Load Models and Vectorizer ===
xgb_model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === 🔥 Local Summarizer Setup ===

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# === 📚 Helper Functions ===

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    return all_text

def summarize_text_local(text, clause_type):
    """Summarizes text using a local HuggingFace model."""
    if len(text) > 1024:
        text = text[:1024]  # BART token limit
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

def predict_document(full_text):
    """Predict clause types and summarize a full document."""
    chunks = full_text.split("\n\n")
    chunks = [c.strip() for c in chunks if len(c.strip()) > 5]  # Filter empty/short chunks

    X_chunks = vectorizer.transform(chunks)
    y_pred = xgb_model.predict(X_chunks)
    predicted_clauses = label_encoder.inverse_transform(y_pred)

    classified_chunks = pd.DataFrame({
        "chunk": chunks,
        "predicted_clause": predicted_clauses
    })

    grouped = classified_chunks.groupby("predicted_clause")["chunk"].apply(lambda x: " ".join(x)).to_dict()

    structured_summaries = {}
    for clause_type, text in grouped.items():
        if len(text) > 5000:
            text = text[:5000]
        summary = summarize_text_local(text, clause_type)
        structured_summaries[clause_type] = summary

    return structured_summaries

def predict_paragraph(text):
    """Predict clause type and summarize a single paragraph."""
    X_paragraph = vectorizer.transform([text])
    y_pred = xgb_model.predict(X_paragraph)
    predicted_clause = label_encoder.inverse_transform(y_pred)[0]

    summary = summarize_text_local(text, predicted_clause)

    return {"predicted_clause": predicted_clause, "summary": summary}

# === 🎨 Streamlit App Interface ===

st.set_page_config(page_title="Contract Clause Classifier & Summarizer", layout="wide")
st.title("📄 Contract Clause Classifier & Summarizer")

tab1, tab2 = st.tabs(["📂 Upload Full Contract", "✏️ Enter Single Paragraph"])

# === 📂 Tab 1: Full Document Upload ===
with tab1:
    uploaded_file = st.file_uploader("Upload a Contract PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_uploaded_contract.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("✅ File uploaded successfully!")
        if st.button("🔍 Analyze Document"):
            full_text = extract_text_from_pdf("temp_uploaded_contract.pdf")
            summaries = predict_document(full_text)

            st.subheader("📑 Summarized Clauses")
            for clause, summary in summaries.items():
                st.markdown(f"### 🗂️ {clause}")
                st.info(summary)

# === ✏️ Tab 2: Single Paragraph Input ===
with tab2:
    user_input = st.text_area("Enter a Contract Paragraph:", height=200)
    
    if st.button("🔍 Analyze Paragraph"):
        if user_input.strip():
            result = predict_paragraph(user_input)
            st.subheader("🔖 Predicted Clause Type")
            st.success(result["predicted_clause"])

            st.subheader("📝 Summarized Paragraph")
            st.info(result["summary"])
        else:
            st.warning("⚠️ Please enter some text to analyze.")