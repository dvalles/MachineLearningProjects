"""
A feedforward no hidden layer neural net
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

input = np.array([1,2])
weights = np.array([1,1])
bias = 2
output = (np.dot(weights,input) + bias)
output = sigmoid(output)
print(output)

