
# Incorporate the mini-batch gradient descent logic into our model training part of the code
# Define number of epochs
n_epochs = 1000

losses = []

# For each epoch ...
for epoch in range(n_epochs):
    # Inner loop: for each mini-batch
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        # The dataset 'lives' in the CPu, so do our mini-batches
        # therefore, we need to send them to the device where the model lives
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Perform one train step and record the corresponding loss
        loss_mini_batch = train_step(x_batch, y_batch)
        mini_batch_losses.append(loss_mini_batch)

    # Compute the average loss for the epoch
    loss = np.mean(mini_batch_losses)
    losses.append(loss)
