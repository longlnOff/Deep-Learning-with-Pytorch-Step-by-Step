
# Defines number of epochs
n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):
    # inner loop
    # Training
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)

    # Validation - NO GRADIENTS IN VALIDATION
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)
