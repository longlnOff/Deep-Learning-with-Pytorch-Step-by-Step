
# our data was in Numpy arrays, but we need to transform them into PyTorch tensors
x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)

# Builds Dataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)

# Builds DataLoader
train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True)
