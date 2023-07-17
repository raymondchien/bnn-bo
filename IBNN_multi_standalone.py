
from models import MultiTaskIBNN
import torch
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Generate example data
n = 100  # Number of training samples
d = 1  # Input dimension
m = 2  # Output dimension

# Generate inputs
train_x = torch.linspace(-5, 5, n).view(-1, d)

# Generate outputs
train_y = torch.stack([
    torch.sin(train_x.squeeze()),
    torch.cos(train_x.squeeze()),
], dim=-1)

# Step 2: Initialize the model
model_args = {"var_b": 1.0, "var_w": 1.0, "depth": 2, "kernel": "ReLU"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskIBNN(model_args, input_dim=d, output_dim=m, device=device)
model = model.to(device)

# Move data to the right device
train_x = train_x.to(device)
train_y = train_y.to(device)

# Step 3: Fit the model
model.fit_and_save(train_x, train_y, "/tmp")

# Step 4: Make predictions
test_x = torch.linspace(-6, 6, 100).view(-1, d).to(device)
posterior = model.posterior(test_x)
mean = posterior.mean.cpu().detach().numpy()
variance = posterior.variance.cpu().detach().numpy()

# Plot the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(test_x.cpu().numpy(), mean[:, 0], label='Predicted')
plt.fill_between(
    test_x.cpu().numpy().squeeze(),
    mean[:, 0] - 1.96 * np.sqrt(variance[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(variance[:, 0]),
    alpha=0.1)
plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy()[:, 0], 'kx', label='Training data')
plt.title("sin(x)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_x.cpu().numpy(), mean[:, 1], label='Predicted')
plt.fill_between(
    test_x.cpu().numpy().squeeze(),
    mean[:, 1] - 1.96 * np.sqrt(variance[:, 1]),
    mean[:, 1] + 1.96 * np.sqrt(variance[:, 1]),
    alpha=0.1)
plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy()[:, 1], 'kx', label='Training data')
plt.title("cos(x)")
plt.legend()

plt.tight_layout()
plt.show()