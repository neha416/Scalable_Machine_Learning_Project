## TITLE:  Real-Time Product Review Quality Predictor

## Project Overview
This project predicts the quality of product reviews in real time using machine learning.  
It demonstrates dynamic data ingestion (new reviews added via API or UI), feature extraction, training, and real-time inference.  
The model predicts whether a review is `high_quality` or `low_quality` based on review text and metadata(rating, helpful votes, verified purchase).



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

## Features

  * Dynamic Data Source:
  - Initial dataset generated from FakeStoreAPI (`create_dataset.py`).  
  - Users can submit new reviews through the Streamlit UI (`app.py`).  
  - Dataset grows dynamically with each new review.

  * Prediction Target: 
  - Review quality: `high_quality` or `low_quality`.

  * Pipelines: 

  - Feature pipeline: TF-IDF for text + metadata features  
  - Training pipeline: Logistic Regression model (`train_model.py`)  
  - Inference pipeline: Real-time prediction through the UI

  * User Interface:
  - Streamlit-based UI for entering reviews  
  - Shows dataset size, predictions, and confidence scores  
  - Supports example predictions for demonstration


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


 ## How to Run

 1. Generate the initial dataset:
     python create_dataset.py

 2. Train the model:
    python train_model.py

 3. Run the Streamlit UI for real-time predictions:
    streamlit run app.py

 4. Submit new reviews through the UI:

 - Enter review text, rating, helpful votes, and verified purchase.

 - Click Submit Review and Predict Quality to append the review to the dataset and get the prediction immediately.

 ## UI LINK: https://scalablemachinelearningproject.streamlit.app