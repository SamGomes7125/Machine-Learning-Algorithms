# K-Means Clustering in R

# Importing the dataset
dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5]

# Using the Elbow Method to find the optimal number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) {
  set.seed(6)
  wcss[i] <- sum(kmeans(X, i)$withinss)
}

# Plot the Elbow Method
plot(x = 1:10, 
     y = wcss, 
     type = 'b', 
     main = 'The Elbow Method', 
     xlab = 'Number of clusters', 
     ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(29)
kmeans_model <- kmeans(x = X, 
                       centers = 5, 
                       iter.max = 300, 
                       nstart = 10)

# Visualizing the Clusters
library(cluster)
clusplot(X, 
         kmeans_model$cluster, 
         lines = 0, 
         shade = TRUE, 
         color = TRUE, 
         labels = 2, 
         plotchar = FALSE, 
         span = TRUE, 
         main = 'Clusters of Customers', 
         xlab = 'Annual Income', 
         ylab = 'Spending Score')
