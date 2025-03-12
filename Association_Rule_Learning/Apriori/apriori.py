#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apriori Algorithm Implementation
Updated for better portability and readability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import os
from apyori import apriori

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Market_Basket_Optimisation.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path, header=None)  # Ensure no header issues
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Market_Basket_Optimisation.csv' is in the same directory.")

# Preparing transaction data
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1]) if str(dataset.values[i, j]) != 'nan'])

# Training Apriori on the dataset
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Converting rules into a readable format
results = []
for rule in rules:
    pair = rule.items_base  # Base items in the rule
    pair2 = rule.items_add  # Items that are recommended
    support = rule.support
    for ordered_stat in rule.ordered_statistics:
        confidence = ordered_stat.confidence
        lift = ordered_stat.lift
        results.append((tuple(pair), tuple(pair2), support, confidence, lift))

# Displaying top 10 results
print("\nTop 10 Association Rules:\n")
for idx, (itemset1, itemset2, support, confidence, lift) in enumerate(results[:10]):
    print(f"Rule {idx+1}: {itemset1} → {itemset2}")
    print(f"   - Support: {support:.4f}, Confidence: {confidence:.4f}, Lift: {lift:.2f}")
    print("-" * 50)
