import streamlit as st
import pandas as pd
import pickle
import numpy as np
from scipy.sparse import hstack

# Load model and vectorizer
model = pickle.load(open("models/review_quality_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("Real-Time Product Review Quality Predictor")

st.write(
    "This application predicts whether a product review is **high quality** "
    "or **low quality** based on review text and metadata."
)


# Dataset Visualization

st.subheader("Dataset Quality Distribution")
df = pd.read_csv("reviews_dataset/review_quality_dataset.csv")
st.bar_chart(df["label"].value_counts())


# User Inputs

st.subheader("Enter Review Details")

review_text = st.text_area("Review Text")

rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
helpful_votes = st.slider("Helpful Votes", 0, 20, 5)
verified_purchase = st.checkbox("Verified Purchase")


# Prediction

if st.button("Predict Review Quality"):
    if review_text.strip() == "":
        st.warning("Please enter review text.")
    else:
        # Vectorize text
        text_vec = vectorizer.transform([review_text])

        # Metadata vector
        meta_vec = np.array([[rating, helpful_votes, int(verified_purchase)]])

        # Combine features
        X = hstack([text_vec, meta_vec])

        # Prediction
        pred_label = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]

        confidence = max(pred_proba)

        if pred_label == "high_quality":
            st.success(f"Prediction: HIGH QUALITY review")
        else:
            st.error(f"Prediction: LOW QUALITY review")

        st.write(f"Confidence Score: **{confidence:.2f}**")


# Example Reviews

st.subheader("Try Example Reviews")

if st.button("Example: Helpful verified review"):
    text_vec = vectorizer.transform(
        ["This product works exactly as described and is very helpful."]
    )
    meta_vec = np.array([[5.0, 10, 1]])
    X = hstack([text_vec, meta_vec])

    pred = model.predict(X)[0]
    proba = max(model.predict_proba(X)[0])

    st.success(f"Prediction: {pred} (Confidence: {proba:.2f})")

if st.button("Example: Low quality review"):
    text_vec = vectorizer.transform(["Bad"])
    meta_vec = np.array([[2.0, 0, 0]])
    X = hstack([text_vec, meta_vec])

    pred = model.predict(X)[0]
    proba = max(model.predict_proba(X)[0])

    st.error(f"Prediction: {pred} (Confidence: {proba:.2f})")



