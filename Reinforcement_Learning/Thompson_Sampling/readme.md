# Thompson Sampling  

Thompson Sampling is a probabilistic algorithm used to solve the multi-armed bandit problem. Unlike UCB, which relies on confidence intervals, Thompson Sampling uses **Bayesian inference** to select the optimal option by sampling from a probability distribution.

## 📌 How It Works  
- Each ad (or action) is modeled as a **Bernoulli distribution**.
- The algorithm **updates the probability distribution** based on past successes and failures.
- Over time, the algorithm **prioritizes the best-performing ads** while still exploring new options.

## 📌 Dataset Used  
- `Ads_CTR_Optimisation.csv` (Simulates click-through rates for different advertisements)

## 📌 Running the Algorithm  
1. **Navigate to the Thompson Sampling subfolder**  
   ```bash
   cd Reinforcement_Learning/Thompson_Sampling
   ```
2. **Run the Python script**  
   ```bash
   python thompson_sampling.py
   ```
3. **View the results showing the optimal ad selections over time.**

## 📌 Requirements  
Ensure you have the following Python libraries installed:  
```bash
pip install numpy pandas matplotlib
