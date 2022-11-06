"""
License

The content of all blog posts (files ending in .md contained in _posts/), all book content (files contained in books/), 
and other content files (files contained in public/files/) remain the sole property of Donald Pinckney, unless otherwise specified. 
The remaining source code used to format and display that content is licensed under the MIT license.

The content of all blog posts (files ending in .md contained in _posts/), all book content (files contained in books/), 
and other content files (files contained in public/files/) remain the sole property of Donald Pinckney, unless otherwise specified. 
The remaining source code used to format and display that content is licensed under the MIT license. In addition, 
any example source code (for example files ending in .py, .ml, etc.) is also licensed under the MIT license.
Source code from https://github.com/donald-pinckney/donald-pinckney.github.io/blob/src/books/pytorch/src/ch2-linreg/code/multi_var_reg/multi_var_reg.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np

### Load the data

# First we load the entire CSV file into numpy
xy = np.loadtxt('content\hackdata.csv', delimiter=",", dtype=np.float32, skiprows=1)
D = torch.from_numpy(xy)

#Get the number of iterations
print("How many iterations do you want to go through?")
try:
    iterations = int(input())
except:
    print("Not a number error")
    exit(0)
# Make a variable to remember the number of input columns
n = 15

# We extract all the input rows for the data
x_dataset = D[:, :n].t()

# We exract our expected output row for the data
y_dataset = D[:, 17].t()

# What we want out
def outputs(iteration, loss, Values, b_val):
    print(f"The best interation is {iteration}")
    print(f"The best loss value is {loss}")
    print(f"The multiple for inlet feed [kg/h] is {Values[0][0]}")
    print(f"The multiple for inlet polymer wt% is {Values[0][1]}")
    print(f"The multiple for inlet A wt% is {Values[0][2]}")
    print(f"The multiple for inlet B wt% is {Values[0][3]}")
    print(f"The multiple for inlet temp [degC] is {Values[0][4]}")
    print(f"The multiple for pressure [MPa] is {Values[0][5]}")
    print(f"The multiple for liquid level [m] is {Values[0][6]}")
    print(f"The multiple for rotate speed [rpm] is {Values[0][7]}")
    print(f"The multiple for bottom temp [degC] is {Values[0][8]}")
    print(f"The multiple for A-Inlet density is {Values[0][9]}")
    print(f"The multiple for A-Outlet Density is {Values[0][10]}")
    print(f"The multiple for B-Inlet Density is {Values[0][11]}")
    print(f"The multiple for B-Outlet Density is {Values[0][12]}")
    print(f"The multiple for Polymer-Inlet Density is {Values[0][13]}")
    print(f"The multiple for Polymer-Outlet Density is {Values[0][14]}")
    print(f"The b value is {b_val}")

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

#Set best variables
best_loss = None
best_iter = 0
best_vals = [[]*n]
best_b_val = 0

# Main optimization loop
for t in range(iterations + 1):
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
    #
    if best_loss == None:
        best_loss = current_loss + 1
    if current_loss < best_loss:
        best_iter = t
        best_vals = A.tolist()
        best_b_val = b.item()
        best_loss=current_loss
outputs(best_iter, best_loss, best_vals, best_b_val)

    