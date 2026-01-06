## TITLE:  Real-Time Product Review Quality Predictor

## Project Overview
This project predicts whether a product review is high quality or low quality using review text and metadata.  
The system performs real-time inference and visualizes results through a Streamlit dashboard.


## Problem Statement
Online marketplaces receive thousands of product reviews, but not all reviews are useful to customers.  
This project aims to automatically identify high-quality reviews using machine learning.

## Prediction Output:
* Binary label: ‘High Quality’ or ‘Low Quality’
* Confidence score (probability)


## Dynamic Data Source
The project uses FakeStoreAPI as a dynamic data source.

* Product data is fetched at runtime from the API
* Review text and metadata are dynamically generated
* Each dataset creation produces new data
* Live user input is processed in real time via the dashboard

This fulfills the requirement of using a dynamic data source.


## Dataset Description
Because FakeStoreAPI does not provide real reviews, synthetic but realistic review data is generated.

## Features:
1. **Text** -Text content of the review
2. **Rating** - Rating given by the user
3. **review_date** - Date when the review was submitted
4. **Verified_purchase** - Indicates whether the purchase is verified
5. **helpful_votes** - Total number of helpful votes
6. **label** - Review quality category (High Quality / Low Quality)

**Labeling Rule**:
A review is labeled High Quality if:
* Rating ≥ 4  
* Helpful votes ≥ 5  
* Verified purchase = True  

Otherwise, it is labeled Low Quality.

## Machine Learning Approach
* Text vectorization using TF-IDF
* Metadata features combined with text features
* Logistic Regression classifier
* Binary classification output


## Technologies Used
* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Streamlit
* Matplotlib



## How to Run the Project
* Install dependencies:
pip install  requirements.txt
* Run the Create Dataset file
python create_dataset.py
and
* Run the application:
streamlit run app.py



