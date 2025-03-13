#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural Language Processing (NLP) - Sentiment Analysis
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Restaurant_Reviews.tsv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path, delimiter='\t', quoting=3)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Restaurant_Reviews.tsv' is in the same directory.")

# Cleaning the text
corpus = []
ps = PorterStemmer()

for i in range(0, len(dataset)):
    review = re.sub(r"[^a-zA-Z]", " ", dataset['Review'][i])  # Remove non-alphabetic characters
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    review = re.sub(r'\s+', ' ', review).strip()  # Remove extra spaces
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values  # Labels (0 = Negative, 1 = Positive)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naïve Bayes to the training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Evaluating model performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Printing the results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(cm)

# Plotting confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
