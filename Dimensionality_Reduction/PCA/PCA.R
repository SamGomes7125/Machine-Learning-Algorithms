# PCA (Principal Component Analysis)

# Importing Dataset
dataset <- read.csv('Wine.csv')

# Splitting Data into Training and Testing Sets
library(caTools)
set.seed(123)  # Ensures reproducibility
split <- sample.split(dataset$Customer_Segment, SplitRatio = 0.80)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling (Only Numeric Columns)
training_set[, -14] <- scale(training_set[, -14])
test_set[, -14] <- scale(test_set[, -14])

# Applying PCA
library(caret)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set_pca = predict(pca, training_set)
training_set_pca = training_set_pca[c(2, 3, 1)]  # Reorder Columns
test_set_pca = predict(pca, test_set)
test_set_pca = test_set_pca[c(2, 3, 1)]  # Reorder Columns

# Ensure Customer_Segment is a factor
training_set_pca$Customer_Segment <- as.factor(training_set$Customer_Segment)
test_set_pca$Customer_Segment <- as.factor(test_set$Customer_Segment)

# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set_pca,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test Set Results
y_pred = predict(classifier, newdata = test_set_pca[-3])

# Creating the Confusion Matrix
cm <- table(test_set_pca$Customer_Segment, y_pred)
print(cm)

# Visualization Function for Training and Test Sets
visualize_results <- function(set, title, classifier) {
  X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  grid_set = expand.grid(X1, X2)
  colnames(grid_set) = c('PC1', 'PC2')
  
  y_grid = predict(classifier, newdata = grid_set)
  
  plot(set[, -3],
       main = title,
       xlab = 'Principal Component 1', ylab = 'Principal Component 2',
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  
  points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', 
                                           ifelse(y_grid == 1, 'springgreen', 'tomato')))
  points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', 
                                    ifelse(set[, 3] == 1, 'green4', 'red3')))
}

# Visualizing the Training Set Results
visualize_results(training_set_pca, "SVM (Training Set with PCA)", classifier)

# Visualizing the Test Set Results
visualize_results(test_set_pca, "SVM (Test Set with PCA)", classifier)
