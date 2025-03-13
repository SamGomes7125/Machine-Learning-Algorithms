#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upper Confidence Bound (UCB) Algorithm Implementation
Optimized for better portability and readability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Ads_CTR_Optimisation.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Ads_CTR_Optimisation.csv' is in the same directory.")

# Implementing UCB Algorithm
N = 10000  # Number of rounds (users)
d = 10  # Number of ads
ads_selected = []
numbers_of_selections = [0] * d  # Count how many times each ad is selected
sums_of_rewards = [0] * d  # Sum of rewards for each ad
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400  # A large number to ensure all ads are selected at least once
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualizing the results
plt.figure(figsize=(10, 5))
plt.hist(ads_selected, bins=np.arange(d+1)-0.5, edgecolor='black', alpha=0.7)
plt.title('Histogram of Ads Selections (UCB)')
plt.xlabel('Ad Index')
plt.ylabel('Number of times each ad was selected')
plt.xticks(range(d))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print(f"Total Reward from UCB Algorithm: {total_reward}")
