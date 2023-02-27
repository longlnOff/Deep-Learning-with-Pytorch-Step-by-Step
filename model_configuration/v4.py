
# Set learning rate
lr = 1e-2

torch.manual_seed(42)

# Builds model
model = nn.Sequential(nn.Linear(1, 1))

# Builds optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Builds loss function
loss_fn = nn.MSELoss(reduction='mean')
