
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set learning rate
lr = 0.001

torch.manual_seed(42)

# Create model and send it to device
model = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(device)

# Defines a optimizer to update the parameters of the model
optimizer = optim.SGD(model.parameters(), lr=lr)

# Define the MSE loss function
loss_fn = torch.nn.MSELoss(reduction='mean')

# Create the train_step function for our model, loss function and optimizer
train_step = make_train_step(model, loss_fn, optimizer)
