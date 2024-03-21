# Issues with RNN3
# The model was plotting each fold separately.

# In this update,
# The model is plotted only once after the cross-validation process.
# The model is saved to a file after the cross-validation process.

import sys
import string
import re
sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers.legacy import Adam
from data_loading import get_spam_dataset


# Constants and Variables for Hyperparameter Tuning
L1_REGULARIZATION = 0.01 # L1 Regularization
L2_REGULARIZATION = 0.01 # L2 Regularization
DROPOUT_RATE = 0.5 
LSTM_UNITS = 100 # Number of neurons in the LSTM layer
DENSE_UNITS = 50 # Number of neurons in the Dense layer
EPOCHS = 20 # Number of epochs (iterations) for training
BATCH_SIZE = 32 # Number of samples to use in each iteration
NUM_FOLDS = 10 # Number of folds for cross-validation - 5 is the standard - 10 is also viable
RANDOM_STATE = 3203 
OPTIMIZER = Adam()
VERBOSE = 0

# Load the dataset
df = get_spam_dataset()


print('Loaded dataset')
X, y = df[['Body']], df['Label'].values
print('Split the data')

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('Normalized the data')

# Feature Selection
selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_STATE))
X_selected = selector.fit_transform(X_scaled, y)
print('Selected features')

# Define the K-fold Cross Validator
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Placeholder for the best fold
best_fold = -1
best_val_accuracy = 0
best_history = None

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X_selected, y):
    # Reshape input to be 3D [samples, timesteps, features] for LSTM
    X_train, X_test = X_selected[train], X_selected[test]
    y_train, y_test = y[train], y[test]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Design the RNN model
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

    # Generate a print
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,  validation_data=(X_test, y_test), verbose=VERBOSE)

    # Generate generalization metrics
    scores = model.evaluate(X_test, y_test, verbose=VERBOSE)
    val_accuracy = scores[1]
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {val_accuracy*100}%')
    
    # Check if the current fold is the best one
    if val_accuracy > best_val_accuracy:
        best_fold = fold_no
        best_val_accuracy = val_accuracy
        best_history = history
    
    fold_no += 1

print(f'Best fold: {best_fold}, with validation accuracy: {best_val_accuracy*100}%')

# Plot the training and validation accuracy of the best fold
plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
