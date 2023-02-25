
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sets Learning Rate
lr = 0.1

torch.manual_seed(42)

# Now we can create a model and send it to the device
model = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(device)


# Define Optimizer
# (Retrieves the parameters of the model and sets the learning rate)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Defines a MSE Loss function
loss_fn = torch.nn.MSELoss()
