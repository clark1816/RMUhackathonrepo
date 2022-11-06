import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np

### Load the data

# First we load the entire CSV file into an m x 3
xy = np.loadtxt('content\hackdata.csv', delimiter=",", dtype=np.float32, skiprows=1)
D = X = torch.from_numpy(xy)

# We extract all rows and the first 2 columns, and then transpose it
x_dataset = D[:, :9].t()

# We extract all rows and the last column, and transpose it
y_dataset = D[:, 11].t()

# And make a convenient variable to remember the number of input columns
n = 9


### Model definition ###

# First we define the trainable parameters A and b 
A = torch.randn((1, n), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Then we define the prediction model
def model(x_input):
    return A.mm(x_input) + b


### Loss function definition ###

def loss(y_predicted, y_target):
    return ((y_predicted - y_target)**2).sum()

### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
optimizer = optim.Adam([A, b], lr=0.1)

# Main optimization loop
for t in range(20000000000):
    # Set the gradients to 0.
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    y_predicted = model(x_dataset)
    # See how far off the prediction is
    current_loss = loss(y_predicted, y_dataset)
    # Compute the gradient of the loss with respect to A and b.
    current_loss.backward()
    # Update A and b accordingly.
    optimizer.step()
    if t % 100000 == 0:
        print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")