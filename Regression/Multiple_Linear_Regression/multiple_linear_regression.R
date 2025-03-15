# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical Data
dataset$State <- factor(dataset$State, 
                        levels = c('New York', 'California', 'Florida'),
                        labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ ., data = training_set)

# Model Summary (Check p-values, R-squared, etc.)
summary(regressor)

# Predicting the Test Results
y_pred = predict(regressor, newdata = test_set)

# Backward Elimination for Feature Selection
library(MASS)
stepwise_model = stepAIC(lm(Profit ~ ., data = dataset), direction = "backward")

# Final Model Summary after elimination
summary(stepwise_model)

# Model Evaluation using Mean Absolute Error (MAE)
mae = mean(abs(y_pred - test_set$Profit))
print(paste("Mean Absolute Error:", round(mae, 2)))
