import requests
import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Create dataset folder if it doesn't exist
os.makedirs("reviews_dataset", exist_ok=True)

# Fetch products from FakeStoreAPI
url = "https://fakestoreapi.com/products"
products = requests.get(url).json()

data = []

# Generate synthetic reviews
for product in products:
    product_title = product["title"]

    # Create 50 reviews per product
    for i in range(50):
        rating = round(random.uniform(1, 5), 1)
        helpful_votes = random.randint(0, 20)
        verified_purchase = random.choice([True, False])
        review_date = datetime.now() - timedelta(days=random.randint(0, 365))

        review_text = (
            f"This is a customer review for {product_title}. "
            f"The product was rated {rating} stars."
        )

        # Review quality rule (GROUND TRUTH)
        if rating >= 4 and helpful_votes >= 5 and verified_purchase:
            label = "high_quality"
        else:
            label = "low_quality"

        data.append({
            "text": review_text,
            "rating": rating,
            "review_date": review_date,
            "verified_purchase": verified_purchase,
            "helpful_votes": helpful_votes,
            "label": label
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("reviews_dataset/review_quality_dataset.csv", index=False)

print("Review quality dataset created successfully!")
print(df.head())
print("\nDataset size:", len(df))
