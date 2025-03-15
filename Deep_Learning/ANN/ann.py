#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:31:42 2025
@author: swarnabhaghosh
"""

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set working directory (Ensure it works universally)
file_path = "Churn_Modelling.csv"

# Load dataset
try:
    dataset = pd.read_csv(file_path)
    print("✅ Data imported successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Error: File not found at {file_path}. Please check the path.")

# Selecting feature columns and target variable
X = dataset.iloc[:, 3:13].values  
y = dataset.iloc[:, 13].values   

# Encoding categorical data (Geography & Gender)
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # Encode 'Geography'
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  # Encode 'Gender'

# One-Hot Encoding 'Geography' column
transformer = ColumnTransformer(transformers=[('one_hot', OneHotEncoder(), [1])], remainder='passthrough')
X = transformer.fit_transform(X)

# Avoiding the Dummy Variable Trap (Removing first column)
X = X[:, 1:]

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Initializing the ANN
classifier = Sequential()

# Input Layer & First Hidden Layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='glorot_uniform', input_shape=(X_train.shape[1],)))
classifier.add(Dropout(0.1))  # Regularization to prevent overfitting

# Second Hidden Layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='glorot_uniform'))
classifier.add(Dropout(0.1))

# Output Layer
classifier.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test))

# Predicting Test set Results
y_pred = (classifier.predict(X_test) > 0.5)

# Model Evaluation
cm = confusion_matrix(y_test, y_pred)
print("\n✅ Confusion Matrix:\n", cm)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
classifier.save("ann_model.h5")
print("\n✅ Model saved successfully as ann_model.h5!")
