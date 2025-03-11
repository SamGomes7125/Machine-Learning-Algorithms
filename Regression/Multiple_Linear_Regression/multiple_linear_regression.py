#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiple Linear Regression Implementation
Updated for better portability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "50_Startups.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure '50_Startups.csv' is in the same directory.")

# Splitting dataset into independent (X) and dependent (y) variables
X = dataset.iloc[:, :-1].values  # Independent variables (R&D Spend, Administration, Marketing Spend, State)
y = dataset.iloc[:, -1].values   # Dependent variable (Profit)

# Encoding categorical data (State column)
column_transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(column_transformer.fit_transform(X))

# Avoiding the Dummy Variable Trap (sklearn does this automatically, so no need to remove one variable)

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression model to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set results
y_pred = regressor.predict(X_test)

# Displaying actual vs predicted values
print("\nActual vs Predicted Profits:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Backward Elimination to find the optimal model
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)  # Adding constant (bias)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]  # Starting with all independent variables

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print("\nOLS Regression Summary (Backward Elimination):")
print(regressor_OLS.summary())
