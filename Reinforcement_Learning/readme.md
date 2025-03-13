# Reinforcement Learning  

Reinforcement Learning (RL) is a type of machine learning where an agent learns by interacting with an environment, receiving rewards, and making decisions to maximize long-term benefits.  

This section includes implementations of the following RL algorithms:

## 📌 1. Upper Confidence Bound (UCB)  
- **Description:** A multi-armed bandit algorithm that balances exploration and exploitation by using confidence intervals to make optimal decisions.  
- **Implementation:** Located in `Reinforcement_Learning/UCB/`  
- **Dataset Used:** `Ads_CTR_Optimisation.csv`  

## 📌 2. Thompson Sampling  
- **Description:** A probabilistic approach to solving the multi-armed bandit problem using Bayesian inference.  
- **Implementation:** Located in `Reinforcement_Learning/Thompson_Sampling/`  
- **Dataset Used:** `Ads_CTR_Optimisation.csv`  

## 📌 How to Use  
Each algorithm is implemented as a Python script inside its respective subfolder. Run the scripts to see how the RL models optimize decision-making.

### Running a Reinforcement Learning Algorithm:
1. **Navigate to the specific algorithm subfolder**  
   ```bash
   cd Reinforcement_Learning/UCB
   ```
2. **Run the Python script**  
   ```bash
   python ucb.py
   ```
3. **View the results showing the optimal ad selections over time.**

## 📌 Requirements  
Ensure you have the following Python libraries installed:  
```bash
pip install numpy pandas matplotlib
