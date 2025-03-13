# Upper Confidence Bound (UCB)  

The Upper Confidence Bound (UCB) algorithm is a reinforcement learning approach for solving the multi-armed bandit problem. It balances **exploration** (trying new options) and **exploitation** (choosing the best-known option) by using confidence intervals to make optimal selections over time.

## 📌 How It Works  
- The algorithm starts by selecting each option once.
- Then, it selects the option with the highest **Upper Confidence Bound** (a combination of past performance and uncertainty).
- Over time, the best option is chosen more frequently while still exploring other possibilities.

## 📌 Dataset Used  
- `Ads_CTR_Optimisation.csv` (Simulates click-through rates for different advertisements)

## 📌 Running the Algorithm  
1. **Navigate to the UCB subfolder**  
   ```bash
   cd Reinforcement_Learning/UCB
   ```
2. **Run the Python script**  
   ```bash
   python ucb.py
   ```
3. **View the results showing which ad was selected most frequently.**

## 📌 Requirements  
Ensure you have the following Python libraries installed:  
```bash
pip install numpy pandas matplotlib
