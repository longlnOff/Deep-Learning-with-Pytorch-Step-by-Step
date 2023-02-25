
torch.manual_seed(13)

# Builds tensors from numpy array BEFORE split
x_tensor = torch.as_tensor(x, dtype=torch.float32)
y_tensor = torch.as_tensor(y, dtype=torch.float32)

# Builds Dataset containing all the data points
dataset = TensorDataset(x_tensor, y_tensor)


# Performs the split
ratio = 0.8
n_total = len(dataset)
n_train = int(ratio * n_total)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# Builds DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)
