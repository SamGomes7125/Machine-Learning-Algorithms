# LDA (Linear Discriminant Analysis)

# Importing Dataset
dataset <- read.csv('Wine.csv')

# Splitting Data into Training and Testing Sets
library(caTools)
set.seed(123)  # Ensures reproducibility
split <- sample.split(dataset$Customer_Segment, SplitRatio = 0.80)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling (Apply Only to Numeric Columns)
training_set[, -14] <- scale(training_set[, -14])
test_set[, -14] <- scale(test_set[, -14])

# Applying LDA
library(MASS)
lda_model = lda(formula = Customer_Segment ~ ., data = training_set)
training_set_lda = as.data.frame(predict(lda_model, training_set))
training_set_lda = training_set_lda[c("x.LD1", "x.LD2", "class")]

test_set_lda = as.data.frame(predict(lda_model, test_set))
test_set_lda = test_set_lda[c("x.LD1", "x.LD2", "class")]

# Convert class labels back to factor
training_set_lda$class <- as.factor(training_set_lda$class)
test_set_lda$class <- as.factor(test_set_lda$class)

# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = class ~ .,
                 data = training_set_lda,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test Set Results
y_pred = predict(classifier, newdata = test_set_lda[-3])

# Creating the Confusion Matrix
cm <- table(test_set_lda$class, y_pred)
print(cm)

# Visualization Function for Training and Test Sets
visualize_results <- function(set, title, classifier) {
  X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  grid_set = expand.grid(X1, X2)
  colnames(grid_set) = c('LD1', 'LD2')
  
  y_grid = predict(classifier, newdata = grid_set)
  
  plot(set[, -3],
       main = title,
       xlab = 'Linear Discriminant 1', ylab = 'Linear Discriminant 2',
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  
  points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', 
                                           ifelse(y_grid == 1, 'springgreen', 'tomato')))
  points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', 
                                    ifelse(set[, 3] == 1, 'green4', 'red3')))
}

# Visualizing the Training Set Results
visualize_results(training_set_lda, "SVM (Training Set with LDA)", classifier)

# Visualizing the Test Set Results
visualize_results(test_set_lda, "SVM (Test Set with LDA)", classifier)
