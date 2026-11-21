
# End-to-End ML Pipeline Object

## Project Overview

This project implements an object-oriented machine learning
pipeline using Python, Pandas and Scikit-learn.

It uses the Ames Housing Dataset to predict house prices
through a Linear Regression model.

## Objective

To develop a reusable ModelTrainer class that initializes
with a raw dataset, cleans the data, prepares training and
testing sets, trains a regression model and evaluates its
performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Google Colab

## Dataset

Ames Housing Dataset obtained through OpenML.

Dataset ID: 42165

Target variable: SalePrice

## Main Features

1. Dataset initialization
2. Duplicate removal
3. Missing value handling
4. Categorical data encoding
5. Training and testing split
6. Linear Regression model training
7. Model evaluation using MSE, RMSE and R2
8. House price prediction
9. Model saving using Joblib

## Project Structure

- model_trainer.py
- main.py
- requirements.txt
- README.md
- End_to_End_ML_Pipeline.ipynb

## Installation

Install the required libraries:

pip install -r requirements.txt

## Execution

Run:

python main.py

Note: The first execution requires an internet
connection to retrieve the dataset from OpenML.

## Machine Learning Algorithm

Linear Regression

## Evaluation Metrics

- Mean Squared Error
- Root Mean Squared Error
- R2 Score

## Conclusion

This project demonstrates how Object-Oriented Programming
can be integrated with Pandas and Scikit-learn to create
a reusable end-to-end machine learning pipeline.
