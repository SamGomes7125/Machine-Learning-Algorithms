#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:04:13 2025

@author: swarnabhaghosh
"""

# XGBoost
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Set working directory dynamically (Portable)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Importing the dataset
file_path = "Churn_Modelling.csv"

try:
    if os.path.isfile(file_path):
        dataset = pd.read_csv(file_path)
        print("Data imported successfully!")
        
        # Selecting features and target
        X = dataset.iloc[:, 3:13].values
        y = dataset.iloc[:, 13].values
    else:
        print(f"File not found: {file_path}")

except FileNotFoundError:
    print(f"Error: File not found: {file_path}") 

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # Encode 'Geography'
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  # Encode 'Gender'

# One-Hot Encoding for Geography
transformer = ColumnTransformer(transformers=[('one_hot', OneHotEncoder(handle_unknown='ignore'), [1])], 
                                remainder='passthrough')
X = transformer.fit_transform(X) 
X = X[:, 1:]  # Avoid dummy variable trap

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the XGBoost Classifier
classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Applying K-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(f"Model Accuracy: {accuracies.mean():.2f} ± {accuracies.std():.2f}")
