#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Tree Regression Implementation
Updated for better portability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor

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
y = dataset.iloc[:, -1].values    # Salary (dependent variable)

# Fitting Decision Tree Regression model to the dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
salary_pred = regressor.predict([[6.5]])
print(f"Predicted Salary using Decision Tree Regression: {salary_pred[0]:.2f}")

# Visualizing Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)  # For smooth curve
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
