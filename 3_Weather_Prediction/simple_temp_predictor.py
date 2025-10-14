# ==============================================================
# simple_temp_predictor.py
# A simple neural network to predict temperature.
# This version:
#   • Uses 500 data points from a CSV file
#   • Does NOT use DataLoader or mini-batches
# ==============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np

# --------------------------------------------------------------
# 1. Load the CSV data
# --------------------------------------------------------------
data = []
with open("3_Weather_Prediction/Data/future_temp_short.csv", newline='') as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)  # skip the first line
    for row in reader:
        # Replace blank entries with 0.0
        cleaned_row = [float(x) if x != "" else 0.0 for x in row]
        data.append(cleaned_row)

data = np.array(data, dtype=np.float32)

# --------------------------------------------------------------
# 2. Split features (X) and target (y)
# --------------------------------------------------------------
X = data[:, :-1]  # all columns except the last
y = data[:, -1:]  # last column is the target

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# --------------------------------------------------------------
# 3. Normalize inputs and target
# --------------------------------------------------------------
X_mean, X_std = X_tensor.mean(0), X_tensor.std(0)
X_tensor = (X_tensor - X_mean) / (X_std + 1e-8)

y_mean, y_std = y_tensor.mean(), y_tensor.std()
y_tensor = (y_tensor - y_mean) / (y_std + 1e-8)

# --------------------------------------------------------------
# 4. Define a simple neural network
# --------------------------------------------------------------
# Tip: Students can change layer sizes or add ReLU for fun!
class TempPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
input_size = X_tensor.shape[1]
model = TempPredictor(input_size)

# --------------------------------------------------------------
# 5. Define loss function and optimizer
# --------------------------------------------------------------
criterion = nn.MSELoss()                    # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam is simple and effective

# --------------------------------------------------------------
# 6. Train the model (no batches, just all data at once)
# --------------------------------------------------------------
epochs = 300
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_tensor)

    # Compute loss
    loss = criterion(y_pred, y_tensor)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")



# --------------------------------------------------------------
# 7. Evaluate and show example predictions
# --------------------------------------------------------------
with torch.no_grad():
    predictions = model(X_tensor) * y_std + y_mean  # un-normalize predictions
    true_values = y_tensor * y_std + y_mean

# Print first few results
for i in range(5):
    print(f"True: {true_values[i].item():.2f}, Predicted: {predictions[i].item():.2f}")


# --------------------------------------------------------------
# 8. Save Your Model
# --------------------------------------------------------------


scripted_model = torch.jit.script(model)
scripted_model.save("3_Weather_Prediction/models/model_0.pth")
print("Full model (architecture + weights) saved successfully!")


# End
print("\nTraining complete!")
