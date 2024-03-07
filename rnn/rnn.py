import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set constants
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
VERBOSE = 1
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

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape input to be 3D [samples, timesteps, features] for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Design the RNN model
model = Sequential([
    LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
