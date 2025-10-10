# train_cos_poly.py
"""
Approximate y = cos(x) using a cubic polynomial:
y = a*x + b*x^2 + c*x^3
Parameters a, b, c are learned by a small neural network using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate training data
x_train = torch.linspace(-2*np.pi, 2*np.pi, 400).unsqueeze(1)  # shape: [200, 1]
y_train = torch.sin(x_train)                                     # shape: [200, 1]

# Define the polynomial model: y = a*x + b*x^2 + c*x^3
class PolyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters initialized randomly
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True))
        self.d = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a * x**1 + self.b * x**2 + self.c * x**3 + self.d * x**4

# Initialize model
model = PolyModel()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 12000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Print learned coefficients
print("\nLearned coefficients:")
print(f"a = {model.a.item():.4f}")
print(f"b = {model.b.item():.4f}")
print(f"c = {model.c.item():.4f}")
print(f"d = {model.c.item():.4f}")

# Plot results
x_plot = x_train.detach().numpy()
y_true = y_train.detach().numpy()
y_pred = model(x_train).detach().numpy()

plt.figure(figsize=(8,5))
plt.plot(x_plot, y_true, label="cos(x)", color='blue')
plt.plot(x_plot, y_pred, label="Polynomial Approximation", color='red', linestyle='--')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Approximating cos(x) with y = a*x + b*x^2 + c*x^3")
plt.show()