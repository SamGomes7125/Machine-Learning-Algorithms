#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Classification Implementation
Updated for better portability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Social_Network_Ads.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Social_Network_Ads.csv' is in the same directory.")

# Splitting dataset into independent (X) and dependent (y) variables
X = dataset.iloc[:, [2, 3]].values  # Selecting Age and Estimated Salary as features
y = dataset.iloc[:, -1].values    # Purchased (0 or 1) as target variable

# Splitting 
