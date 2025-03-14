# Simple Linear Regression in R

# Importing the dataset
dataset <- read.csv('Salary_Data.csv')

# Splitting the dataset into Training and Test set
library(caTools)
set.seed(123)  # Set seed for reproducibility
split <- sample.split(dataset$Salary, SplitRatio = 2/3) 
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Fitting Simple Linear Regression to the Training Set
regressor = lm(Salary ~ YearsExperience, data = training_set)

# Predicting the Test Results
y_pred = predict(regressor, newdata = test_set)

# Model Summary
summary(regressor)  # Displays R-squared & p-values

# Visualising the Training Set Results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualising the Test Set Results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Evaluating Model Performance
r_squared <- summary(regressor)$r.squared
print(paste("R-squared:", round(r_squared, 3)))
