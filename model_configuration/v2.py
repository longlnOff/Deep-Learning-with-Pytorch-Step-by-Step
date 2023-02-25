
# Set the learning rate
learning_rate = 1e-3

torch.manual_seed(42)

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(1, 1)).to(device)

# Define optimizer to update the model's parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Defines a MSE loss function
loss_fn = torch.nn.MSELoss()

# Create the train-step function for our model, loss function and optimizer
train_step = make_train_step(model, loss_fn, optimizer)

# Create the validation-step function for our model and loss function
val_step = make_val_step(model, loss_fn)
