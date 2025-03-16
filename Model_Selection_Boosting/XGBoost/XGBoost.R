# XGBoost Classification

# Load Required Libraries
library(caTools)
library(caret)
library(xgboost)

# Import Dataset
dataset <- read.csv('Churn_Modelling.csv')
dataset <- dataset[, 4:14]

# Encoding Categorical Variables as Factors
dataset$Geography <- as.numeric(factor(dataset$Geography, levels = c('France', 'Spain', 'Germany'), labels = c(1, 2, 3)))
dataset$Gender <- as.numeric(factor(dataset$Gender, levels = c('Female', 'Male'), labels = c(1, 2)))

# Splitting Dataset into Training and Test Sets
set.seed(123)
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Converting Data to XGBoost Matrix Format
dtrain <- xgb.DMatrix(data = as.matrix(training_set[, -11]), label = training_set$Exited)
dtest <- xgb.DMatrix(data = as.matrix(test_set[, -11]))

# Training XGBoost Model
classifier <- xgboost(data = dtrain, nrounds = 50, objective = "binary:logistic")

# Predicting Test Set Results
y_pred <- predict(classifier, newdata = dtest)
y_pred <- as.numeric(y_pred >= 0.5)

# Confusion Matrix
cm <- table(test_set$Exited, y_pred)
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
folds <- createFolds(training_set$Exited, k = 10)
cv_results <- lapply(folds, function(x) {
  training_fold <- training_set[-x, ]
  test_fold <- training_set[x, ]
  
  dtrain_fold <- xgb.DMatrix(data = as.matrix(training_fold[, -11]), label = training_fold$Exited)
  dtest_fold <- xgb.DMatrix(data = as.matrix(test_fold[, -11]))
  
  classifier <- xgboost(data = dtrain_fold, nrounds = 50, objective = "binary:logistic")
  
  y_pred_fold <- predict(classifier, newdata = dtest_fold)
  y_pred_fold <- as.numeric(y_pred_fold >= 0.5)
  
  cm_fold <- table(test_fold[, 11], y_pred_fold)
  accuracy_fold <- sum(diag(cm_fold)) / sum(cm_fold)
  return(accuracy_fold)
})

cv_accuracy <- mean(unlist(cv_results))
print(paste("Cross-Validation Accuracy:", round(cv_accuracy * 100, 2), "%"))
