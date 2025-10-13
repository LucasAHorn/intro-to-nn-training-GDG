import torch
import torch.nn as nn
import csv
import numpy as np

# ===============================
# 1. Load the trained model (TorchScript)
# ===============================
model = torch.jit.load("3_Weather_Prediction/models/model_0.pth")
model.eval()  # disable dropout, batchnorm, etc.

# ===============================
# 2. Load the dataset
# ===============================
data_list = []
with open("3_Weather_Prediction/Data/future_temp_short.csv", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # skip header
    for row in csv_reader:
        cleaned_row = [float(x) if x != '' else 0.0 for x in row]
        data_list.append(cleaned_row)

data_array = np.array(data_list, dtype=np.float32)

# Split features (X) and target (y)
X = data_array[:, :-1]
y = data_array[:, -1:]

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# ===============================
# 3. Normalize inputs and target
# ===============================
# IMPORTANT: Use same normalization as training
# If you saved mean/std during training, load them here instead of recomputing
X_mean, X_std = X_tensor.mean(0), X_tensor.std(0)
y_mean, y_std = y_tensor.mean(), y_tensor.std(0)

X_norm = (X_tensor - X_mean) / (X_std + 1e-8)

# ===============================
# 4. Evaluate the model on the full dataset
# ===============================
with torch.no_grad():
    y_pred_norm = model(X_norm)
    y_pred = y_pred_norm * y_std + y_mean  # convert back to original scale

# ===============================
# 5. Compute overall errors
# ===============================
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()  # mean absolute error (average error)

mse_value = mse_loss(y_pred, y_tensor).item()
avg_error = mae_loss(y_pred, y_tensor).item()

# ===============================
# 6. Print results
# ===============================
print(f"\tEvaluation complete on {len(X_tensor)} samples")
print(f"Mean Squared Error (MSE): {mse_value:.6f}")
print(f"Average Error (MAE): {avg_error:.6f}")

# Print first few sample predictions
for i in range(min(5, len(X_tensor))):
    print(f"Sample {i+1}: True = {y_tensor[i].item():.3f}, Pred = {y_pred[i].item():.3f}")