import requests
import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Create dataset directory if it doesn't exist
os.makedirs("reviews_dataset", exist_ok=True)

DATASET_PATH = "reviews_dataset/review_quality_dataset.csv"

# Fetch product data dynamically at runtime
API_URL = "https://fakestoreapi.com/products"
products = requests.get(API_URL).json()

new_records = []

# Generate dynamic reviews for each product
for product in products:
    product_title = product["title"]

    # Generate a small batch each run (simulates live incoming data)
    for _ in range(10):
        rating = round(random.uniform(1, 5), 1)
        helpful_votes = random.randint(0, 20)
        verified_purchase = random.choice([True, False])
        review_date = datetime.now() - timedelta(days=random.randint(0, 30))

        review_text = (
            f"This is a live customer review for {product_title}. "
            f"The product received a rating of {rating} stars."
        )

        # Label generation (ground truth logic)
        label = (
            "high_quality"
            if rating >= 4 and helpful_votes >= 5 and verified_purchase
            else "low_quality"
        )

        new_records.append({
            "text": review_text,
            "rating": rating,
            "review_date": review_date,
            "verified_purchase": verified_purchase,
            "helpful_votes": helpful_votes,
            "label": label
        })

# Convert new batch to DataFrame
df_new = pd.DataFrame(new_records)

# Append to existing dataset if it exists
if os.path.exists(DATASET_PATH):
    df_existing = pd.read_csv(DATASET_PATH)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new

# Save updated dataset
df_final.to_csv(DATASET_PATH, index=False)

print("Dynamic data ingestion completed successfully.")
print("New records added:", len(df_new))
print("Total dataset size:", len(df_final))

