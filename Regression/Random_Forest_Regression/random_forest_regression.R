# Random Forest Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]  # Keep only Level and Salary columns

# Loading necessary library
library(randomForest)
set.seed(1234)

# Fitting Random Forest Regression to the dataset
regressor = randomForest(x = dataset[, 1, drop = FALSE],  # Ensure it's a DataFrame
                         y = dataset$Salary,
                         ntree = 500)

# Predicting a new result with Random Forest Regression
y_pred = predict(regressor, data.frame(Level = 6.5))
print(paste("Predicted Salary for Level 6.5:", round(y_pred, 2)))

# Model Summary
print(regressor)  # Displays number of trees, MSE, etc.
importance(regressor)  # Shows feature importance

# Visualising the Random Forest Regression results (higher resolution)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')

# Compute Mean Squared Error (MSE)
mse = mean((dataset$Salary - predict(regressor, dataset))^2)
print(paste("Mean Squared Error:", round(mse, 2)))

# Plotting the tree structure (for small tree sizes)
plot(regressor)
text(regressor)
