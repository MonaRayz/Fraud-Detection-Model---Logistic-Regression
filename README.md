# Fraud-Detection-Model---Logistic-Regression
Applying Machine Learning in Detection of Credit Card Fraud
This project implements a logistic regression model to detect potentially fraudulent transactions. The model is trained on transactional data and uses several key features to predict the likelihood that a given transaction is fraudulent.

Project Structure
data/: Contains the transaction dataset (not included due to privacy).
notebooks/: Jupyter notebooks for data analysis, preprocessing, and model training.
scripts/: Python scripts for data preprocessing, model training, evaluation, and deployment.
README.md: Project overview and usage instructions.
Model Overview
Problem Statement
The goal is to classify transactions as fraudulent or non-fraudulent based on a set of transaction characteristics. This binary classification problem helps financial institutions identify suspicious activities and prevent fraudulent transactions.

Features Used
The model leverages the following features to determine the likelihood of fraud:

Amount: The monetary amount of the transaction.
Transfer Type: The type of transfer (e.g., payment or cash out).
Transaction Purpose: Indicates if the transaction is a payment or cash-out transaction.
Account Difference: The difference in account numbers or identifiers between the origin and destination banks.
Target Variable
Fraudulent: A binary variable indicating whether the transaction is fraudulent (1) or not (0).
