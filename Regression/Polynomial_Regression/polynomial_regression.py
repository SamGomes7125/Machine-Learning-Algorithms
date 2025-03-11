#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polynomial Regression Implementation
Updated for better portability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Position_Salaries.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Position_Salaries.csv' is in the same directory.")

# Splitting dataset into independent (X) and dependent (y) variables
X = dataset.iloc[:, 1:-1].values  # Position Level (independent variable)
y = dataset.iloc[:, -1].values   # Salary (dependent variable)

# Fitting Linear Regression to the dataset for comparison
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=4)  # Degree 4 for better curve fitting
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
salary_pred_linear = lin_reg.predict([[6.5]])
print(f"Predicted Salary using Linear Regression: {salary_pred_linear[0]:.2f}")

# Predicting a new result with Polynomial Regression
salary_pred_poly = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(f"Predicted Salary using Polynomial Regression: {salary_pred_poly[0]:.2f}")
