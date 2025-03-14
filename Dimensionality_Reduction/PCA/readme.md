# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a technique used to reduce the number of dimensions in a dataset while retaining as much variance as possible.

## 📌 Applications  
- Feature reduction in **classification and regression models**  
- Helps in **visualizing high-dimensional data**  
- Used in **image compression and face recognition**  

## 📌 How PCA Works  
- Standardizes the dataset  
- Computes the **covariance matrix**  
- Calculates **eigenvectors & eigenvalues**  
- Selects **top principal components**  

## 📌 Running the PCA Algorithm  
1. **Navigate to the PCA folder**  
   ```bash
   cd Dimensionality_Reduction/PCA
   ```
2. **Run the PCA script**  
   ```bash
   python pca.py
   ```

## 📌 Requirements  
```bash
pip install numpy pandas matplotlib sklearn
