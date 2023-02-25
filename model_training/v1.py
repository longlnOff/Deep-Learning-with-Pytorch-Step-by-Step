
# Define number of epochs
n_epochs = 1000

losses = [] 

# For each epoch ...
for epoch in range(n_epochs):
    # Perform one train step and record the corresponding loss
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)
