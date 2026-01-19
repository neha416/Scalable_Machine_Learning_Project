import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from datetime import datetime
from scipy.sparse import hstack

# ----------------------------
# Configuration
# ----------------------------
DATASET_PATH = "reviews_dataset/review_quality_dataset.csv"

# ----------------------------
# Load model and vectorizer
# ----------------------------
model = pickle.load(open("models/review_quality_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ----------------------------
# Helper: Dynamic data ingestion
# ----------------------------
def ingest_new_review(review_text, rating, helpful_votes, verified_purchase):
    """
    Dynamically ingests a new review into the dataset (Lab 1 requirement)
    """
    new_record = {
        "text": review_text,
        "rating": rating,
        "helpful_votes": helpful_votes,
        "verified_purchase": verified_purchase,
        "review_date": datetime.now(),
        "label": None  # label unknown at ingestion time
    }

    new_df = pd.DataFrame([new_record])

    if os.path.exists(DATASET_PATH):
        existing_df = pd.read_csv(DATASET_PATH)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    updated_df.to_csv(DATASET_PATH, index=False)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Real-Time Product Review Quality Predictor")

st.write(
    "This application dynamically ingests new product reviews and predicts "
    "whether a review is **high quality** or **low quality** using machine learning."
)

# ----------------------------
# Dataset Visualization (Backfill + Ingested Data)
# ----------------------------
if os.path.exists(DATASET_PATH):
    st.subheader("Dataset Size (Backfill + Dynamic Ingestion)")
    df = pd.read_csv(DATASET_PATH)
    st.write(f"Total reviews in dataset: **{len(df)}**")

# ----------------------------
# User Input (Dynamic Data Source)
# ----------------------------
st.subheader("Enter a New Product Review")

review_text = st.text_area("Review Text")
rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
helpful_votes = st.slider("Helpful Votes", 0, 20, 5)
verified_purchase = st.checkbox("Verified Purchase")

# ----------------------------
# Prediction + Ingestion Pipeline
# ----------------------------
if st.button("Submit Review and Predict Quality"):
    if review_text.strip() == "":
        st.warning("Please enter review text.")
    else:
        # 1️⃣ Dynamic data ingestion (Lab 1 requirement)
        ingest_new_review(
            review_text,
            rating,
            helpful_votes,
            int(verified_purchase)
        )

        # 2️⃣ Feature pipeline (same as training)
        text_vec = vectorizer.transform([review_text])
        meta_vec = np.array([[rating, helpful_votes, int(verified_purchase)]])
        X = hstack([text_vec, meta_vec])

        # 3️⃣ Inference pipeline
        pred_label = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        confidence = max(pred_proba)

        if pred_label == "high_quality":
            st.success("Prediction: HIGH QUALITY review")
        else:
            st.error("Prediction: LOW QUALITY review")

        st.write(f"Confidence Score: **{confidence:.2f}**")

# ----------------------------
# Example Reviews (Optional)
# ----------------------------
st.subheader("Example Predictions")

if st.button("Example: High Quality Review"):
    example_text = "This product works exactly as described and is very helpful."
    text_vec = vectorizer.transform([example_text])
    meta_vec = np.array([[5.0, 10, 1]])
    X = hstack([text_vec, meta_vec])
    pred = model.predict(X)[0]
    proba = max(model.predict_proba(X)[0])
    st.success(f"Prediction: {pred} (Confidence: {proba:.2f})")

if st.button("Example: Low Quality Review"):
    example_text = "Bad"
    text_vec = vectorizer.transform([example_text])
    meta_vec = np.array([[2.0, 0, 0]])
    X = hstack([text_vec, meta_vec])
    pred = model.predict(X)[0]
    proba = max(model.predict_proba(X)[0])
    st.error(f"Prediction: {pred} (Confidence: {proba:.2f})")




