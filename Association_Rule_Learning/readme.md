# Association Rule Learning (ARL)

Association Rule Learning is an unsupervised machine learning technique used to discover interesting relationships, or "rules," between variables in large datasets. It is commonly applied in market basket analysis, where businesses identify product associations based on customer purchasing behavior.

This folder contains implementations of the two main association rule learning algorithms:

## 📌 1. Apriori Algorithm  
- **Description:** Apriori is an algorithm that finds frequent item sets in a transactional dataset and derives association rules based on confidence, support, and lift.  
- **Key Features:** Uses a breadth-first search strategy and the "Apriori property" to prune infrequent itemsets.  
- **Implementation:** Located in `Association_Rule_Learning/Apriori/`  
- **Dataset Used:** `Market_Basket_Optimisation.csv`  

## 📌 2. Eclat Algorithm (Coming Next)  
- **Description:** Eclat (Equivalence Class Clustering and Bottom-Up Lattice Traversal) is an alternative to Apriori that uses a depth-first search strategy for faster rule mining.  
- **Key Features:** More memory efficient as it uses intersection-based set operations instead of candidate generation.  
- **Implementation:** Located in `Association_Rule_Learning/Eclat/`  
- **Dataset Used:** `Market_Basket_Optimisation.csv`  

## 📌 How to Use  
Each algorithm has its own Python script and dataset inside its respective subfolder. You can run the scripts to generate association rules.

📌 Running the Algorithms
1️⃣ Navigate to the ARL folder
bash
Copy
Edit
cd ARL
2️⃣ Run the Python script for the association rule learning method
bash
Copy
Edit
python apriori.py  # Runs the Apriori Algorithm
python eclat.py    # Runs the Eclat Algorithm
3️⃣ Observe the results
Apriori identifies frequent itemsets and generates association rules.
Eclat efficiently finds frequent patterns using depth-first search.
📌 Requirements
Ensure you have the following Python libraries installed before running the scripts:

bash
Copy
Edit
pip install numpy pandas apyori
✉️ Contributions & Issues
Feel free to submit a pull request if you have improvements.
If you find any issues, create a GitHub issue describing the problem.
