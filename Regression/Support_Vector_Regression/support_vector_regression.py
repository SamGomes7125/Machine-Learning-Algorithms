#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support Vector Regression (SVR) Implementation
Updated for better portability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
y = dataset.iloc[:, -1].values.reshape(-1, 1)  # Salary (dependent variable, reshaped for scaling)

# Feature Scaling (SVR requires feature scaling)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

# Fitting SVR model to the dataset
regressor = SVR(kernel='rbf')  # Using Radial Basis Function (RBF) kernel
regressor.fit(X_scaled, y_scaled.ravel())

# Predicting a new result
salary_pred_scaled = regressor.predict(sc_X.transform([[6.5]]))
salary_pred = sc_y.inverse_transform(salary_pred_scaled.reshape(-1, 1))

print(f"Predicted Salary using Support Vector Regression: {salary_pred[0][0]:.2f}")

# Visualizing SVR results
plt.scatter(X_scaled, y_scaled, color='red')
plt.plot(X_scaled, regressor.predict(X_scaled), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary (Scaled)')
plt.show()
