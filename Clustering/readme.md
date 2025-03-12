# Clustering Algorithms

Clustering is an unsupervised machine learning technique used to group data points into clusters based on their similarities. Unlike classification, clustering does not rely on predefined labels but instead identifies patterns and structures within the dataset.

This folder contains implementations of different clustering algorithms:

## 📌 1. K-Means Clustering
- **Description:** K-Means is a centroid-based clustering algorithm that partitions data into `K` distinct clusters.
- **Key Features:** Uses the Elbow Method to determine the optimal number of clusters and visualizes the clusters along with their centroids.
- **Implementation:** Located in `Clustering/K_Means/`
- **Dataset Used:** `Mall_Customers.csv`

## 📌 2. Hierarchical Clustering (Coming Next)
- **Description:** Hierarchical clustering builds a tree-like structure (dendrogram) to determine cluster relationships.
- **Key Features:** Can be **Agglomerative** (bottom-up approach) or **Divisive** (top-down approach).
- **Implementation:** Located in `Clustering/Hierarchical/`
- **Dataset Used:** `Mall_Customers.csv`

## 📌 How to Use
Each clustering algorithm has its own Python script and dataset inside its respective subfolder. You can run the scripts to see how the data is grouped into clusters.

### Running a Clustering Algorithm:
1. **Navigate to the specific clustering subfolder**  
   ```bash
   cd Clustering/K_Means
