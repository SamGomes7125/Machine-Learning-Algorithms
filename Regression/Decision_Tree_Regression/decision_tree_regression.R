# Decision Tree Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]  # Selecting relevant columns

# Load necessary libraries
library(rpart)
library(ggplot2)

# Fitting Decision Tree Regression to the dataset
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 2))

# Predicting a new result with Decision Tree Regression
y_pred = predict(regressor, data.frame(Level = 6.5))
print(paste("Predicted Salary for Level 6.5:", y_pred))

# Visualising the Decision Tree Regression results (higher resolution)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary') +
  theme_minimal()

# Plotting the Decision Tree
plot(regressor)
text(regressor, use.n = TRUE, all = TRUE, cex = 0.8)

# Saving the plot (optional)
ggsave("decision_tree_plot.png")
