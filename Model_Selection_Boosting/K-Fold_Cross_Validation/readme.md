# K-Fold Cross Validation  

K-Fold Cross Validation is a technique used to evaluate the performance of machine learning models by splitting the dataset into **K equal parts (folds)**. The model is trained on **K-1 folds** and tested on the remaining **1 fold**. This process is repeated **K times**, with each fold serving as the test set once.  

## 📌 Why Use K-Fold Cross Validation?  
- **More reliable model evaluation** compared to train-test splits.  
- **Prevents overfitting** by ensuring all data points are used for training and testing.  
- **Helps in selecting the best hyperparameters** when used with Grid Search.  

## 📌 How It Works  
1️⃣ Split the dataset into **K folds** (e.g., `K=10`).  
2️⃣ Train the model on `K-1` folds and test it on the remaining fold.  
3️⃣ Repeat this process `K` times, each time using a different fold for testing.  
4️⃣ Compute the **average accuracy** and **standard deviation** across all folds.  

## 📌 Running the Algorithm  
1️⃣ Navigate to the **K-Fold Cross Validation** folder:  
   ```bash
   cd K-Fold_Cross_Validation
2️⃣ Run the Python script:

bash
Copy
Edit
python k_fold.py
3️⃣ Observe the mean accuracy and standard deviation of the model performance.

📌 Requirements
Ensure you have the following installed:

bash
Copy
Edit
pip install numpy pandas matplotlib scikit-learn
✉️ For issues or contributions, feel free to submit a pull request! 🚀


