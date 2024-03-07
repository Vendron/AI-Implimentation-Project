# Issues with RNN2
# The model was overfitting the data.

# In this update, 
# L1 and L2 regularization are added to the model.
# Cross Validation is implemented to prevent overfitting.
# Dropout layers are used to prevent overfitting.

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

# Constants and Variables for Hyperparameter Tuning
L1_REGULARIZATION = 0.01 # L1 Regularization
L2_REGULARIZATION = 0.01 # L2 Regularization
DROPOUT_RATE = 0.5 
LSTM_UNITS = 100 # Number of neurons in the LSTM layer
DENSE_UNITS = 50 # Number of neurons in the Dense layer
EPOCHS = 20 # Number of epochs (iterations) for training
BATCH_SIZE = 32 # Number of samples to use in each iteration
NUM_FOLDS = 5 # Number of folds for cross-validation - 5 is the standard - 10 is also viable
RANDOM_STATE = 3203 
OPTIMIZER = Adam()

# Load the dataset
df = pd.read_csv('../data/spam.csv')
print('Loaded dataset')

# Split the data
X = df.iloc[:, 1:-1].values  # Exclude the first and last column
y = df.iloc[:, -1].values
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
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,  validation_data=(X_test, y_test), verbose=0) # Verbose=1 shows the progress bar, 

    # Generate generalization metrics
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    fold_no += 1

    # == Plot Training and Validation Loss for each fold ==
    plt.plot(history.history['loss'], label=f'Training Loss for fold {fold_no}')
    plt.plot(history.history['val_loss'], label=f'Validation Loss for fold {fold_no}')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()