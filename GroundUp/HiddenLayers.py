"""
A dense network with several layers, activation functions, and a softmax output
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDeriv(x):
    return np.exp(-x)/((np.exp(-x)+1)**2)

def layer(inputSize, layerWidth):
    return np.random.randn(layerWidth,inputSize)

def softmax(arr):
    total = np.sum(np.exp(arr))
    softmaxEl = lambda el : np.exp(el)/total
    return softmaxEl(arr)

#SETUP NETWORK

input = np.random.randn(20)

h1 = layer(input.shape[0]+1, 25)

h2 = layer(h1.shape[0]+1, 10)

h3 = layer(h2.shape[0]+1, 25)

out = layer(h3.shape[0]+1, 5)

#RUN NETWORK
input = np.append(input, [1]) #bias
out1 = np.dot(h1,input) #multiply
act1 = sigmoid(out1) #activate

act1 = np.append(act1, [1]) #bias
out2 = np.dot(h2,act1) #multiply
act2 = sigmoid(out2) #activate

act2 = np.append(act2, [1]) #bias
out3 = np.dot(h3,act2) #multiply
act3 = sigmoid(out3) #activate

act3 = np.append(act3, [1]) #bias
out4 = np.dot(out,act3) #multiply
outputRaw = sigmoid(out4) #activate

output = softmax(outputRaw)

print(output)





