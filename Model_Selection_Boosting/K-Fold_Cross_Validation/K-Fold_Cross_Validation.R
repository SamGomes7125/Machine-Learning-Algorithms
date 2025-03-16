# Kernel SVM Classification

# Load Required Libraries
library(caTools)
library(e1071)
library(caret)
library(ggplot2)

# Import Dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset <- dataset[, 3:5]

# Encoding Target Variable as Factor
dataset$Purchased <- factor(dataset$Purchased, levels = c(0, 1))

# Splitting Dataset into Training and Test Sets
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[, -3] <- scale(training_set[, -3])
test_set[, -3] <- scale(test_set[, -3])

# Training Kernel SVM Model
classifier <- svm(formula = Purchased ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'radial')

# Predicting Test Set Results
y_pred <- predict(classifier, newdata = test_set[-3])

# Confusion Matrix
cm <- table(test_set[, 3], y_pred)
print(cm)

# Performance Metrics
precision <- cm[2,2] / (cm[2,2] + cm[1,2])
recall <- cm[2,2] / (cm[2,2] + cm[2,1])
f1_score <- 2 * ((precision * recall) / (precision + recall))

print(paste("Precision:", round(precision, 2)))
print(paste("Recall:", round(recall, 2)))
print(paste("F1-Score:", round(f1_score, 2)))

# Applying K-Fold Cross-Validation
set.seed(123)
folds <- createFolds(training_set$Purchased, k = 10)
cv_results <- lapply(folds, function(x) {
  training_fold <- training_set[-x, ]
  test_fold <- training_set[x, ]
  
  classifier <- svm(formula = Purchased ~ .,
                    data = training_fold,
                    type = 'C-classification',
                    kernel = 'radial')
  
  y_pred_fold <- predict(classifier, newdata = test_fold[-3])
  cm_fold <- table(test_fold[, 3], y_pred_fold)
  accuracy_fold <- sum(diag(cm_fold)) / sum(cm_fold)
  return(accuracy_fold)
})
cv_accuracy <- mean(unlist(cv_results))
print(paste("Cross-Validation Accuracy:", round(cv_accuracy * 100, 2), "%"))

# Function for Visualization
visualize_results <- function(set, title, classifier) {
  X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  grid_set <- expand.grid(X1, X2)
  colnames(grid_set) <- c('Age', 'EstimatedSalary')
  
  y_grid <- predict(classifier, newdata = grid_set)
  
  plot(set[, -3],
       main = title,
       xlab = 'Age', ylab = 'Estimated Salary',
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'dodgerblue', 'salmon'))
  points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'dodgerblue3', 'salmon3'))
}

# Visualizing Training Set Results
visualize_results(training_set, "Kernel SVM (Training Set)", classifier)

# Visualizing Test Set Results
visualize_results(test_set, "Kernel SVM (Test Set)", classifier)
