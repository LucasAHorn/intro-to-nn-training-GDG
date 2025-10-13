# ============================================================
# train_taylor_series.py
# Approximate y = cos(x) using a cubic polynomial:
#     y = a*x + b*x^2 + c*x^3
# We'll learn the coefficients (a, b, c) using PyTorch.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. Set random seed (so results are reproducible)
# ------------------------------------------------------------
torch.manual_seed(0)

# ------------------------------------------------------------
# 2. Generate training data
# ------------------------------------------------------------
# TODO: Change sin to cos later if we want to match the title!
x_train = torch.linspace(-2 * np.pi, 2 * np.pi, 200).unsqueeze(1)
y_train = torch.____(x_train)     # TODO: fill in the function (sin then cos)

# ------------------------------------------------------------
# 3. Define the polynomial model
# ------------------------------------------------------------
# y = a*x + b*x^2 + c*x^3
class PolyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each parameter starts as a random value, then is trained
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True))
        # You can try more layouts if you desire
        # self.d = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        # TODO: write the equation for the cubic polynomial
        # Hint: return self.a * x + self.b * x**2 + self.c * x**3
        return self.a * x + self.b * x**2 + self.c * x**3

# ------------------------------------------------------------
# 4. Initialize model, loss, and optimizer
# ------------------------------------------------------------
model = PolyModel()

criterion = nn._____()   # TODO: fill in the loss (MSELoss)
optimizer = optim._____(model.parameters(), lr=0.01)  # TODO: pick optimizer (Adam or SGD)

# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------
num_epochs = 2000

for epoch in range(num_epochs):
    # Step 1: reset gradients
    optimizer.zero_grad()

    # Step 2: forward pass (predict y)
    y_pred = model(x_train)

    # Step 3: compute loss
    loss = criterion(y_pred, y_train)

    # Step 4: backward pass
    loss.backward()

    # Step 5: update parameters
    optimizer.step()

    # Print progress every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# ------------------------------------------------------------
# 6. Print learned coefficients
# ------------------------------------------------------------
print("\nLearned coefficients:")
print(f"a = {model.a.item():.4f}")
print(f"b = {model.b.item():.4f}")
print(f"c = {model.c.item():.4f}")

# ------------------------------------------------------------
# 7. Plot results
# ------------------------------------------------------------
x_plot = x_train.detach().numpy()
y_true = y_train.detach().numpy()
y_pred = model(x_train).detach().numpy()

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_true, label="True Function", color="blue")
plt.plot(x_plot, y_pred, label="Polynomial Approximation", color="red", linestyle="--")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Approximating y = cos(x) with a cubic polynomial")
plt.show()