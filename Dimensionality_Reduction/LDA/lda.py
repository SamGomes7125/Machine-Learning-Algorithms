#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Discriminant Analysis (LDA) - Dimensionality Reduction
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Wine.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Wine.csv' is in the same directory.")

# Splitting dataset into features (X) and target (y)
X = dataset.iloc[:, 0:13].values  # Selecting first 13 columns (Chemical properties)
y = dataset.iloc[:, 13].values    # Target column (Wine class)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling (Required for LDA)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA - Choosing top 2 Linear Discriminants
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Training Logistic Regression on LDA-transformed data
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_lda, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test_lda)

# Evaluating model performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Printing the results
print(f"Model Accuracy after LDA: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(cm)

# Plotting confusion matrix using Seaborn
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix (After LDA)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualizing the LDA results (Training Set)
plt.figure(figsize=(7,5))
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap="coolwarm", edgecolor="k")
plt.title("LDA Applied to Wine Dataset (Training Set)")
plt.xlabel("Linear Discriminant 1 (LD1)")
plt.ylabel("Linear Discriminant 2 (LD2)")
plt.colorbar()
plt.show()
