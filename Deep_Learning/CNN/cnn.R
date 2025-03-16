# Convolutional Neural Network (CNN) in R using Keras & TensorFlow

# Install and Load Required Packages
if (!require(keras)) install.packages("keras")
if (!require(tensorflow)) install.packages("tensorflow")

library(keras)
library(tensorflow)

# Install TensorFlow (if not already installed)
install_tensorflow()

# Data Preprocessing
train_datagen <- image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

training_set <- flow_images_from_directory(
  "dataset/training_set",
  train_datagen,
  target_size = c(64, 64),
  batch_size = 32,
  class_mode = "binary"
)

test_set <- flow_images_from_directory(
  "dataset/test_set",
  test_datagen,
  target_size = c(64, 64),
  batch_size = 32,
  class_mode = "binary"
)

# Building the CNN Model
cnn_model <- keras_model_sequential() %>%
  
  # Convolution Layer 1
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(64, 64, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Convolution Layer 2
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Flattening
  layer_flatten() %>%
  
  # Fully Connected Layer
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  
  # Output Layer (Binary Classification)
  layer_dense(units = 1, activation = "sigmoid")

# Compile the CNN Model
cnn_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Train the CNN Model
history <- cnn_model %>% fit(
  training_set,
  steps_per_epoch = 250,
  epochs = 25,
  validation_data = test_set,
  validation_steps = 63
)

# Save the Model
save_model_hdf5(cnn_model, "cnn_model.h5")

# Print Summary
summary(cnn_model)
