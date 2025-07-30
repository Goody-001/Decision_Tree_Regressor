# Second-Hand Car Price Prediction using Decision Tree Regressor

This project demonstrates how machine learning can be used to predict the selling price of second-hand cars based on various features such as fuel type, mileage, transmission type, engine size, and more. The goal is to help users estimate a fair price for used vehicles by training a regression model on a real-world dataset.

## Project Overview
Dataset: Contains detailed records of used cars including price, specifications, and features.

Model: Decision Tree Regressor from Scikit-learn.

Goal: Predict car prices and reduce error through model optimization.

## Techniques:

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Model training and evaluation

Hyperparameter tuning using GridSearchCV

## Technologies Used
Python

Pandas – for data manipulation

NumPy – for numerical operations

Matplotlib & Seaborn – for data visualization

Scikit-learn – for model training and evaluation

## Project Workflow
Data Loading & Inspection

Read the dataset

Check data types, missing values, and summary statistics

Data Preprocessing

Drop irrelevant columns 

Define features X and target y

Exploratory Data Analysis (EDA)

Correlation heatmap

Model Training

Split data into train and test sets

Train a Decision Tree Regressor

Evaluate using metrics like RMSE and R² score

Hyperparameter Tuning

Use GridSearchCV to search for the best parameters

Compare performance before and after tuning

## Results
Root Mean Squared Error (RMSE): 43,981.43

R² Score: 0.8713

The model performance improved significantly after hyperparameter tuning, demonstrating the importance of selecting optimal parameters.

## Key Learnings
Decision Trees can effectively model non-linear relationships in pricing data.

Hyperparameter tuning (e.g., max_depth, min_samples_split) helps reduce overfitting and improve generalization.

# Author
Goodluck Nwachukwu
Learning. Building. Growing.
