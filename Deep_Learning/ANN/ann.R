# Artificial Neural Network (ANN) in R using H2O

# Load required packages
if (!require(h2o)) install.packages("h2o", dependencies=TRUE)
library(h2o)

# Import the dataset
dataset <- read.csv('Churn_Modelling.csv')
dataset <- dataset[4:14]  # Selecting relevant columns

# Encoding categorical variables as factors
dataset$Geography <- as.numeric(factor(dataset$Geography, levels = c('France', 'Spain', 'Germany'), labels = c(1, 2, 3)))
dataset$Gender <- as.numeric(factor(dataset$Gender, levels = c('Female', 'Male'), labels = c(1, 2)))

# Splitting dataset into Training and Test sets
library(caTools)
set.seed(123)
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] <- scale(training_set[-11])
test_set[-11] <- scale(test_set[-11])

# Initialize H2O
h2o.init(nthreads = -1, max_mem_size = "2G")

# Convert datasets to H2O format
training_h2o <- as.h2o(training_set)
test_h2o <- as.h2o(test_set[-11])  # Exclude target column

# Train the ANN model
model <- h2o.deeplearning(
  y = "Exited", 
  training_frame = training_h2o,
  activation = "Rectifier",  # ReLU Activation
  hidden = c(6, 6),  # 2 hidden layers with 6 neurons each
  epochs = 100,
  train_samples_per_iteration = -2,
  standardize = TRUE
)

# Predicting on the Test set
y_pred <- h2o.predict(model, newdata = test_h2o)
y_pred <- as.vector(y_pred$predict > 0.5)  # Convert H2O frame to vector

# Confusion Matrix
cm <- table(test_set[, 11], y_pred)
print("Confusion Matrix:")
print(cm)

# Shutdown H2O after model execution
h2o.shutdown(prompt = FALSE)
