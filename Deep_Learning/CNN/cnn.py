#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:46:26 2025
@author: swarnabhaghosh
"""

# Convolutional Neural Network (CNN)
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Automatically detect script directory (portable)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding 2nd convolutional layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Fully Connected Layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
training_set = train_datagen.flow_from_directory(os.path.join(script_dir, 'dataset/training_set'), target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(os.path.join(script_dir, 'dataset/test_set'), target_size=(64, 64), batch_size=32, class_mode='binary')

# Train the CNN
classifier.fit(training_set, steps_per_epoch=len(training_set), epochs=25, validation_data=test_set, validation_steps=len(test_set))

# Save the model
classifier.save(os.path.join(script_dir, 'cnn_model.h5'))  # Save model in the same directory
print("✅ CNN Model Training Completed!")
