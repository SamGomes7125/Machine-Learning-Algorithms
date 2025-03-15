# Support Vector Regression (SVR) in R

# Importing the dataset
dataset <- read.csv('Position_Salaries.csv')
dataset <- dataset[, 2:3]  # Selecting relevant columns

# Feature Scaling (Required for SVR)
dataset[, 1:2] <- scale(dataset[, 1:2])

# Fitting SVR Model to the Dataset
library(e1071)
regressor <- svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')

# Predicting a new result with SVR
new_level <- scale(data.frame(Level = 6.5))  # Scale input to match training
y_pred <- predict(regressor, new_level)

# Visualizing the SVR Results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') + 
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')

# Print Prediction
print(paste("Predicted Salary for Level 6.5:", round(y_pred, 2)))
