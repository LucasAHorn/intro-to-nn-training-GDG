import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import csv
import numpy as np

# =========================
# 1. Load CSV data
# =========================
data_list = []
with open("Completed_Models/3_Weather_Prediction/Data/future_temp_year.csv", newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # skip header
    for row in csv_reader:
        cleaned_row = [float(x) if x != '' else 0.0 for x in row]
        data_list.append(cleaned_row)

data_array = np.array(data_list, dtype=np.float32)

# =========================
# 2. Split features and target
# =========================
X = data_array[:, :-1]
y = data_array[:, -1:]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# =========================
# 3. Normalize inputs and target
# =========================
X_mean, X_std = X_tensor.mean(0), X_tensor.std(0)
X_tensor = (X_tensor - X_mean) / (X_std + 1e-8)

y_mean, y_std = y_tensor.mean(), y_tensor.std()
y_tensor = (y_tensor - y_mean) / (y_std + 1e-8)

# =========================
# 4. Create Dataset and DataLoader (train/val split)
# =========================
dataset = TensorDataset(X_tensor, y_tensor)
val_size = int(0.1 * len(dataset))  # 10% validation
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# =========================
# 5. Define neural network
# =========================
class TempPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TempPredictor(input_size=X_tensor.shape[1]).to(device)

# =========================
# 6. Loss and optimizer
# =========================
criterion = nn.SmoothL1Loss()  # more robust than MSE
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Adaptive scheduler: reduce LR when validation loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",       # minimize loss
    factor=0.5,       # shrink LR by half
    patience=10,      # wait 10 epochs without improvement
    threshold=1e-4,   # consider significant improvement
    cooldown=0,
    min_lr=1e-9,      # don’t go below this LR
    eps=1e-8
)

# =========================
# 7. Training loop
# =========================
epochs = 200
best_val_loss = float("inf")
patience, wait = 30, 0  # early stopping

for epoch in range(epochs):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_ds)

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(device), yv.to(device)
            pv = model(Xv)
            val_loss += criterion(pv, yv).item() * Xv.size(0)
    val_loss /= len(val_ds)

    # Step scheduler
    scheduler.step(val_loss)

    # Print progress
    if (epoch + 1) % (epochs // 20) == 0:
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.9f}"
        )

    # Early stopping + checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.jit.script(model).save("Completed_Models/3_Weather_Prediction/models/model_0.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break