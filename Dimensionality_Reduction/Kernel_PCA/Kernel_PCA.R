# Kernel PCA with Logistic Regression

# Load Libraries
library(caret)
library(kernlab)
library(caTools)
library(ggplot2)

# Import Dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset <- dataset[, 3:5]

# Splitting Dataset into Training and Testing Sets
set.seed(123)  # Ensures reproducibility
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 1:2] <- scale(training_set[, 1:2])
test_set[, 1:2] <- scale(test_set[, 1:2])

# Applying Kernel PCA
kpca <- kpca(~., data = training_set[, 1:2], kernel = "rbfdot", features = 2)
train_pca <- as.data.frame(predict(kpca, training_set[, 1:2]))
test_pca <- as.data.frame(predict(kpca, test_set[, 1:2]))

# Renaming Features for Clarity
colnames(train_pca) <- c("KPCA1", "KPCA2")
colnames(test_pca) <- c("KPCA1", "KPCA2")

# Adding Target Variable
train_pca$Purchased <- training_set$Purchased
test_pca$Purchased <- test_set$Purchased

# Fitting Logistic Regression
classifier <- glm(Purchased ~ ., data = train_pca, family = binomial)

# Predicting Test Set Results
y_pred <- ifelse(predict(classifier, test_pca, type = "response") > 0.5, 1, 0)

# Creating Confusion Matrix
cm <- table(test_pca$Purchased, y_pred)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Visualization Function
visualize_results <- function(set, title, classifier) {
  X1 <- seq(min(set$KPCA1) - 1, max(set$KPCA1) + 1, by = 0.01)
  X2 <- seq(min(set$KPCA2) - 1, max(set$KPCA2) + 1, by = 0.01)
  grid_set <- expand.grid(KPCA1 = X1, KPCA2 = X2)
  
  prob_set <- predict(classifier, newdata = grid_set, type = 'response')
  y_grid <- ifelse(prob_set > 0.5, 1, 0)
  
  plot(set[, -3],
       main = title,
       xlab = 'Kernel PCA 1', ylab = 'Kernel PCA 2',
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  
  points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen', 'tomato'))
  points(set, pch = 21, bg = ifelse(set$Purchased == 1, 'green4', 'red3'))
}

# Visualizing Training Set Results
visualize_results(train_pca, "Logistic Regression (Training Set with KPCA)", classifier)

# Visualizing Test Set Results
visualize_results(test_pca, "Logistic Regression (Test Set with KPCA)", classifier)
