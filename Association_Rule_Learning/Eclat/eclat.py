#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eclat Algorithm Implementation
Updated for better portability and readability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import os
from itertools import combinations

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

# Function to generate frequent itemsets
def get_frequent_itemsets(transactions, min_support=0.003):
    item_counts = {}
    total_transactions = len(transactions)

    # Count frequency of individual items
    for transaction in transactions:
        for item in set(transaction):
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1

    # Filter by minimum support
    frequent_items = {item: count / total_transactions for item, count in item_counts.items() if (count / total_transactions) >= min_support}

    return frequent_items

# Function to generate association rules
def generate_eclat_rules(transactions, min_support=0.003):
    frequent_items = get_frequent_itemsets(transactions, min_support)
    itemsets = list(frequent_items.keys())
    itemset_supports = {}

    # Generate frequent itemsets of size 2 and above
    for size in range(2, len(itemsets) + 1):
        for itemset in combinations(itemsets, size):
            support = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction))) / len(transactions)
            if support >= min_support:
                itemset_supports[itemset] = support

    return itemset_supports

# Running Eclat
eclat_rules = generate_eclat_rules(transactions, min_support=0.003)

# Displaying top 10 frequent itemsets
print("\nTop 10 Frequent Itemsets (Eclat Algorithm):\n")
for idx, (itemset, support) in enumerate(sorted(eclat_rules.items(), key=lambda x: x[1], reverse=True)[:10]):
    print(f"Itemset {idx+1}: {itemset} (Support: {support:.4f})")
    print("-" * 50)
