# Naive Bayes Classification in R

# Importing the dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset <- dataset[, 3:5]  # Selecting relevant columns

# Encoding the target feature as factor
dataset$Purchased <- factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into Training and Test sets
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling (only for numerical variables)
training_set[, 1:2] <- scale(training_set[, 1:2])
test_set[, 1:2] <- scale(test_set[, 1:2])

# Fitting Naive Bayes to the Training set
library(e1071)
classifier <- naiveBayes(x = training_set[-3], y = training_set$Purchased)

# Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm <- table(test_set$Purchased, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy Calculation
accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Visualizing the Training set results
library(ElemStatLearn)
set <- training_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, length.out = 100)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, length.out = 100)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- predict(classifier, newdata = grid_set)

plot(set[, -3], main ='Naive Bayes (Training Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3')) 

# Visualizing the Test set results
set <- test_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, length.out = 100)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, length.out = 100)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- predict(classifier, newdata = grid_set)

plot(set[, -3], main ='Naive Bayes (Test Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3')) 
