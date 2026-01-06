import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import os

# Create model folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("reviews_dataset/review_quality_dataset.csv")

# Features and label
X_text = df["text"]
X_meta = df[["rating", "helpful_votes", "verified_purchase"]]
y = df["label"]

# Convert boolean to int
X_meta["verified_purchase"] = X_meta["verified_purchase"].astype(int)

# Train-test split
X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_text, X_meta, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF for text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec = vectorizer.transform(X_text_test)

# Combine text + metadata
X_train = hstack([X_text_train_vec, X_meta_train.values])
X_test = hstack([X_text_test_vec, X_meta_test.values])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, preds))

# Save model and vectorizer
pickle.dump(model, open("models/review_quality_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\nModel trained and saved successfully!")
