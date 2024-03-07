import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Function to load data from CSV file
def load_data(filepath):
    # Read the dataset
    df = pd.read_csv(filepath)
    # Extract features and labels, assuming the first column is an identifier and the last one is the target
    X = df.iloc[:, 1:-1].values  
    y = df.iloc[:, -1].values
    return X, y

# Function to preprocess data
def preprocess_data(X, y):
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the SpamRNN class extending the nn.Module class
class SpamRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(SpamRNN, self).__init__()
        # Initialize the RNN layer(s)
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)
        # Initialize the fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    # Define the forward pass
    def forward(self, x):
        # Pass the input through the RNN layer(s)
        out, _ = self.rnn(x)
        # Get the output of the last timestep and pass it through the fully connected layer
        out = self.fc(out[:, -1, :])  
        return out

# Function to train the model
def train_model(filepath):
    print("Starting training...")
    X_train, X_test, y_train, y_test = preprocess_data(*load_data(filepath))
    
    # Reshape data to add a sequence dimension
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Adding a feature dimension
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Set model parameters
    input_dim = 1  # Each timestep has 1 feature
    hidden_dim = 128
    output_dim = 2  # Binary classification
    n_layers = 2

    model = SpamRNN(input_dim, hidden_dim, output_dim, n_layers)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Iterate through epochs
    for epoch in range(10):
        optimizer.zero_grad()  # Clear gradients for this training step
        output = model(X_train_tensor)  # Forward pass
        loss = loss_fn(output, y_train_tensor)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Apply optimization
        print(f'Epoch {epoch+1}/{10}, Loss: {loss.item()}')  # Print loss for the current epoch

if __name__ == "__main__":
    train_model('../data/spam.csv')
