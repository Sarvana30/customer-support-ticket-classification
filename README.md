# Customer Support Ticket Classification and Priority Prediction

## Project Overview

This project focuses on building a machine learning-based customer support system that can classify customer support tickets, predict their priority, and retrieve a relevant response.

The system takes a customer's support query as input and performs three main tasks:

1. **Intent Classification** – identifies the intent behind the customer's query.
2. **Priority Prediction** – predicts whether the query has High, Medium, or Low priority.
3. **Response Retrieval** – retrieves the most relevant response from the existing customer support dataset.

## Project Workflow

Customer Query  
↓  
Intent Classification  
↓  
Priority Prediction  
↓  
Response Retrieval  
↓  
Relevant Customer Support Response

## Machine Learning Models

The following classification models were trained and evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

These models were used for both intent classification and priority prediction.

## Response Retrieval

Response retrieval is implemented using:

- TF-IDF Vectorization
- Cosine Similarity

The system compares a new customer query with existing customer instructions and retrieves the response associated with the most similar query.

## Dataset

The dataset contains customer support information including:

- `instruction`
- `response`
- `category`
- `intent`
- `priority`

The priority labels were created based on the customer intent.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF
- Cosine Similarity
- Jupyter Notebook / Google Colab

## Project Modules

The notebooks in this project cover:

1. Dataset Exploration
2. Data Cleaning
3. Text Preprocessing
4. Intent Classification
5. Priority Prediction
6. Response Retrieval
7. End-to-End Customer Support System

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries:

```bash
pip install -r requirements.txt