# Create a SingleTaskIBNN model
import torch

from models import SingleTaskIBNN

model_args = {
    "var_b": 1.0,
    "var_w": 0.1,
    "depth": 3,
    "kernel": "erf"
}
input_dim = 1
output_dim = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SingleTaskIBNN(model_args, input_dim, output_dim, device)

# Fit the model to training data
train_x = torch.tensor([[0.0], [1.0], [2.0]])
train_y = torch.tensor([[0.0], [1.0], [2.0]])
model.fit_and_save(train_x, train_y, save_dir="model_checkpoint.pt")

# Perform posterior inference on test data
test_x = torch.tensor([[3.0], [4.0]])
posterior = model.posterior(test_x)

# Get mean and variance predictions
mean = posterior.mean  # Shape: (2, 1)
variance = posterior.variance  # Shape: (2, 1)
