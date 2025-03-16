# Hierarchical Clustering

# Importing Dataset
dataset <- read.csv('mall.csv') 
X = dataset[4:5]  # Selecting Annual Income and Spending Score

# Compute the distance matrix only once
distance_matrix <- dist(X, method = 'euclidean')

# Using Dendrogram to find the optimal number of clusters
dendrogram = hclust(distance_matrix, method = 'ward.D')

# Enhanced visualization with color and titles
plot(dendrogram,
     main = "Dendrogram for Optimal Clusters",
     xlab = "Number of Customers", 
     ylab = "Euclidean Distance",
     sub = "",
     cex = 0.9,  # Adjust font size
     hang = -1,  # Align labels at the bottom
     col = "blue")  

# Applying Hierarchical Clustering
hc = hclust(distance_matrix, method = 'ward.D')
optimal_clusters = 5  # You can dynamically determine this from the dendrogram
y_hc = cutree(hc, optimal_clusters)

# Visualizing the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = "Clusters of Customers",
         xlab = "Annual Income",
         ylab = "Spending Score",
         col.p = y_hc,  # Different colors for different clusters
         cex = 0.9)  # Adjust point size
