# Polynomial Regression

# Importing Dataset
library(readr)
library(ggplot2)
dataset <- read_csv('Position_Salaries.csv') 
dataset = dataset[, 2:3]  # Keep only Level and Salary columns

# Fitting Simple Linear Regression
lin_reg = lm(formula = Salary ~ Level, data = dataset)

# Fitting Polynomial Regression using poly()
poly_reg = lm(formula = Salary ~ poly(Level, 4, raw = TRUE), data = dataset)

# Model Summaries
summary(lin_reg)
summary(poly_reg)

# Visualising Linear Regression Results
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth/Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising Polynomial Regression Results (Higher Resolution)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid))), colour = 'blue') +
  ggtitle('Truth/Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
y_pred2 = predict(poly_reg, data.frame(Level = 6.5))

# Model Evaluation (Mean Squared Error)
mse_lin = mean((dataset$Salary - predict(lin_reg, dataset))^2)
mse_poly = mean((dataset$Salary - predict(poly_reg, dataset))^2)

print(paste("Linear Regression MSE:", round(mse_lin, 2)))
print(paste("Polynomial Regression MSE:", round(mse_poly, 2)))
