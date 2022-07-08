"""
Implementation of the delta rule for no hidden layer feedforward networks
No activation function
"""

import numpy as np

#training set [inputX, inputY, biasInput, intendedOutput]
training = [
    [np.random.uniform(1,2), np.random.uniform(1,2), 1, -1],
    [np.random.uniform(1,2), np.random.uniform(1,2), 1, -1],
    [np.random.uniform(1,2), np.random.uniform(1,2), 1, -1],
    [np.random.uniform(4,5), np.random.uniform(-1,-2), 1, 1],
    [np.random.uniform(4,5), np.random.uniform(-1,-2), 1, 1],
    [np.random.uniform(4,5), np.random.uniform(-1,-2), 1,  1]
]

learningRate = .03
weights = np.array([40,-300,0], dtype='float64') #weights and bias

for x in range(30000):
    #training sample
    sample = np.random.randint(0,6)
    input = training[sample][0:3]
    label = training[sample][-1:]

    #inference
    output = (np.dot(weights,input))

    #error
    error = (label[0] - output)
    #error gradients
    gradOne  = -1*input[0]*error
    gradTwo  = -1*input[1]*error
    gradBias = -1*input[2]*error
    #readjust weights
    grads = np.array([gradOne, gradTwo, gradBias], dtype='float64')
    grads = grads*learningRate
    grads = grads * -1
    weights += grads

    if x%3000 == 0:
        print(weights)


