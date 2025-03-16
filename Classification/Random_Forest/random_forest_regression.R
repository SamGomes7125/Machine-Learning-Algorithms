# Random Forest Regression in R

# Importing the dataset
dataset <- read.csv('Position_Salaries.csv', stringsAsFactors = TRUE)
dataset <- dataset[, 2:3]  # Selecting only relevant columns

# Fitting Random Forest Regression to the dataset
library(randomForest)
set.seed(1234)  # Ensuring reproducibility

# Training the Random Forest Model
regressor <- randomForest(x = dataset["Level"],
                          y = dataset$Salary,
                          ntree = 500)

# Predicting a new result with Random Forest Regression
y_pred <- predict(regressor, data.frame(Level = 6.5))
print(paste("Predicted Salary for Level 6.5:", round(y_pred, 2)))

# Model Evaluation (R² and RMSE)
y_actual <- dataset$Salary
y_fitted <- predict(regressor, dataset["Level"])

r_squared <- 1 - sum((y_actual - y_fitted)^2) / sum((y_actual - mean(y_actual))^2)
rmse <- sqrt(mean((y_actual - y_fitted)^2))

print(paste("R² Score:", round(r_squared, 3)))
print(paste("Root Mean Squared Error (RMSE):", round(rmse, 2)))

# Visualizing the Random Forest Regression results (higher resolution)
library(ggplot2)
x_grid <- seq(min(dataset$Level), max(dataset$Level), length.out = 100)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')

# Plotting the tree structure
plot(regressor, main = "Random Forest Decision Trees")
text(regressor)
