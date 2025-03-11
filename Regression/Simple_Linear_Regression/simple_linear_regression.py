#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Linear Regression Implementation
Updated for better portability
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Salary_Data.csv")

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Salary_Data.csv' is in the same directory.")

# Splitting dataset into independent (X) and dependent (y) variables
X = dataset.iloc[:, :-1].values  # Years of Experience
y = dataset.iloc[:, -1].values   # Salary

# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting the Simple Linear Regression model to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualizing the Training Set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test Set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Same regression line
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
