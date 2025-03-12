#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means Clustering Implementation
Updated for better portability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Mall_Customers.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Mall_Customers.csv' is in the same directory.")

# Selecting relevant features for clustering
X = dataset.iloc[:, [3, 4]].values  # Using Annual Income and Spending Score

# Using the Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Applying K-Means with the optimal number of clusters (K=5)
kmeans = KMeans(n_clusters=5, init='k-means++', max
