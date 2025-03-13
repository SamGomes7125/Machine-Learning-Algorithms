#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thompson Sampling Algorithm Implementation
Optimized for better portability and readability
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Ads_CTR_Optimisation.csv")  # Dataset file

# Checking if the dataset exists
if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
    print("Data imported successfully!")  
else:
    raise FileNotFoundError(f"Error: File not found: {file_path}. Please make sure 'Ads_CTR_Optimisation.csv' is in the same directory.")

# Implementing Thompson Sampling Algorithm
N = 10000  # Number of rounds (users)
d = 10  # Number of ads
ads_selected = []
numbers_of_rewards_1 = [0] * d  # Number of times ad i got reward 1
numbers_of_rewards_0 = [0] * d  # Number of times ad i got reward 0
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rew
