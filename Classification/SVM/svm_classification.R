# Support Vector Machine (SVM) Classification

# Importing the dataset
dataset <- read.csv('Social_Network_Ads.csv', stringsAsFactors = TRUE)
dataset <- dataset[, 3:5]  # Selecting relevant features

# Splitting the dataset into Training and Test sets
library(caTools)
set.seed(123)  # Ensuring reproducibility
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 1:2] <- scale(training_set[, 1:2])
test_set[, 1:2] <- scale(test_set[, 1:2])

# Fitting SVM to the Training set
library(e1071)
classifier <- svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm <- table(test_set$Purchased, y_pred)
print("Confusion Matrix:")
print(cm)

# Model Evaluation (Accuracy, Precision, Recall, F1-Score)
accuracy <- sum(diag(cm)) / sum(cm)
precision <- cm[2,2] / sum(cm[,2])
recall <- cm[2,2] / sum(cm[2,])
f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("Accuracy:", round(accuracy, 3)))
print(paste("Precision:", round(precision, 3)))
print(paste("Recall:", round(recall, 3)))
print(paste("F1-Score:", round(f1_score, 3)))

# Visualizing the Training set results
library(ggplot2)
set <- training_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- predict(classifier, newdata = grid_set)

ggplot() +
  geom_point(aes(x = set$Age, y = set$EstimatedSalary, color = factor(set$Purchased))) +
  geom_contour(aes(x = grid_set$Age, y = grid_set$EstimatedSalary, z = as.numeric(y_grid)), bins = 1, color = "black") +
  ggtitle('SVM (Training Set)') +
  xlab('Age') +
  ylab('Estimated Salary') +
  scale_color_manual(values = c('red', 'green')) +
  theme_minimal()

# Visualizing the Test set results
set <- test_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- predict(classifier, newdata = grid_set)

ggplot() +
  geom_point(aes(x = set$Age, y = set$EstimatedSalary, color = factor(set$Purchased))) +
  geom_contour(aes(x = grid_set$Age, y = grid_set$EstimatedSalary, z = as.numeric(y_grid)), bins = 1, color = "black") +
  ggtitle('SVM (Test Set)') +
  xlab('Age') +
  ylab('Estimated Salary') +
  scale_color_manual(values = c('red', 'green')) +
  theme_minimal()
