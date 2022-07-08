"""
A dense network with several layers, and activation functions
Trained on XOR function using SGD
"""

import numpy as np
import math

def sigmoid(x): #verified correct
    return 1/(1+np.exp(-x))

def sigmoidDeriv(x): #verified correct
    return np.exp(-x)/((np.exp(-x)+1)**2)

def layer(inputSize, layerWidth):
    return np.random.randn(layerWidth,inputSize)

#TRAINING SET linear classifier [inputX, inputY, intendedOutput]
trainingLinear = [
    [np.random.uniform(1,2), np.random.uniform(1,2), -1],
    [np.random.uniform(1,2), np.random.uniform(1,2), -1],
    [np.random.uniform(1,2), np.random.uniform(1,2), -1],
    [np.random.uniform(4,5), np.random.uniform(-1,-2), 1],
    [np.random.uniform(4,5), np.random.uniform(-1,-2), 1],
    [np.random.uniform(4,5), np.random.uniform(-1,-2), 1]
]

#TRAINING SET [0,5] sum of sines
trainingSin = np.ndarray([501,2])
i = 0
for x in np.arange(0,5.01,.01):
    trainingSin[i] = np.array([x,math.sin(x)])
    i += 1

#TRAINING SET xor function
trainingxor = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
# trainingxor = np.array([[1,1,.5]])

#SETUP NETWORK

input = np.random.randn(2) #xor

w1 = layer(input.shape[0]+1, 4)

w2 = layer(w1.shape[0]+1, 1)

lr = 6e-3
trainingSamples = 20000000
#TRAINING STOCHASITC GRADIENT
for t in range(0,trainingSamples):

    #index training
    index = t%trainingxor.shape[0]
    if index == 0:
        np.random.shuffle(trainingxor)

    #sampling
    sample = trainingxor[index]
    input = sample[:2]
    label = sample[2]

    #Feedforward
    input = np.append(input, [1]) #bias
    z1 = np.dot(w1,input) #multiply
    y1 = sigmoid(z1) #activate

    y1 = np.append(y1, [1]) #bias
    z2 = np.dot(w2,y1) #multiply
    output = sigmoid(z2) #activate

    #Backprop
    #output activations local gradient
    outputLocalGrads = np.ndarray([output.shape[0]])
    for x in range(0, output.shape[0]):
        #localGrad = dE/de * de/dy * dy/dz
        #dE/de E = 1/2*Σ(e)^2 dE/de = e
        error = output[x] - label
        #de/dy e = (y-t) de/dy = 1
        dedy = 1
        #dy/dz dy = sigmoid(z) dy/dz = e^(-z)/(e^(-z)+1)^2
        dydz = sigmoidDeriv(z2[x])
        #local grad
        locGrad = error*dedy*dydz
        outputLocalGrads[x] = locGrad

    #grads for w2
    #dz/dw dz = Σwi*xi + b dz/dw = xi
    w2Grads = w2.copy()
    for y in range(0, y1.shape[0]): #for each neuron in previous layer
        for o in range(0, output.shape[0]): #for each neuron in next layer
            w2Grads[o,y] = -lr*y1[y]*outputLocalGrads[o]

    #partial derivs for 1st layer net acts, sum all partials
    z1Derivs = np.zeros(z1.shape)
    for z in range(0, z1.shape[0]): #for each neuron in previous layer
        for o in range(0, output.shape[0]): #for each neuron in next layer
            z1Derivs[z] += w2[o,z]*outputLocalGrads[o]
        z1Derivs[z] = z1Derivs[z]*sigmoidDeriv(z1[z])

    #grads for w1
    w1Grads = w1.copy()
    for i in range(0, input.shape[0]): #for each neuron in previous layer
        for z in range(0, z1.shape[0]): #for each neuron in next layer
            w1Grads[z,i] = -lr*input[i]*z1Derivs[z]

    #UPDATE LAYERS
    w1 += w1Grads
    w2 += w2Grads

    if t % 10000 == 0:
        print(input, output, error)













