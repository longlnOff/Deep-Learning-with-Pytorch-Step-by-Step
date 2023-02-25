import torch

# Define number of epochs
epochs = 100
for epoch in range(epochs):
    # Set model to TRAIN mode
    model.train()

    # Compute model's predictions
    y_pred = model(x_train_tensor)

    # Compute the loss
    loss = loss_fn(y_pred, y_train_tensor)

    # Compute the gradient
    loss.backward()

    # Update the parameters
    optimizer.step()
    optimizer.zero_grad()
