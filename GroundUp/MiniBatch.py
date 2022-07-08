"""
A dense network with several layers, and activation functions
Trained on several datasets with mini batch SGD
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
    trainingSin[i] = np.array([x,math.sin(x)*.5+.5])
    i += 1

#TRAINING SET xor function
trainingxor = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
# trainingxor = np.array([[1,1,.5]])

#SETUP NETWORK

input = np.random.randn(1) #sin function

w1 = layer(input.shape[0]+1, 20)

w2 = layer(w1.shape[0]+1, 20)

w3 = layer(w2.shape[0]+1, 20)

w4 = layer(w3.shape[0]+1, 1)

#run a forward pass of network
def inference(input, label):
    #first layer
    input = np.append(input, [1]) #bias
    z1 = np.dot(w1,input) #multiply
    y1 = sigmoid(z1) #activate

    #second layer
    y1 = np.append(y1, [1]) #bias
    z2 = np.dot(w2,y1) #multiply
    y2 = sigmoid(z2) #activate

    #third layer
    y2 = np.append(y2, [1]) #bias
    z3 = np.dot(w3,y2) #multiply
    y3 = sigmoid(z3) #activate

    #fourth layer
    y3 = np.append(y3, [1]) #bias
    z4 = np.dot(w4,y3) #multiply
    output = sigmoid(z4) #activate
    
    error = output - label
    return input, z1, y1, z2, y2, z3, y3, z4, output, error

#region Back Prop Functions

#local grads for last layer (batch)
def OutputLocalGradsBatch(batch, inputs, outputs, errors):
        outputLocalGradsBatch = np.ndarray(outputs.shape)
        for s in range(0, len(batch)):
            outputLocalGradsBatch[s] = OutputLocalGrads(s, inputs, outputs, errors)
        return outputLocalGradsBatch

#Local grads for last layer
def OutputLocalGrads(s, inputs, outputs, errors):
    outputLocalGrads = np.ndarray(outputs[s].shape[0])
    for x in range(0, outputs[s].shape[0]):
        #dE/de E = 1/2*Σ(e)^2 dE/de = e
        error = errors[s][x] #so this was wrong
        #de/dy e = (y-t) de/dy = 1
        dedy = 1
        #dy/dz dy = sigmoid(z) dy/dz = e^(-z)/(e^(-z)+1)^2
        dydz = sigmoidDeriv(inputs[s][x])
        #localGrad = dE/de * de/dy * dy/dz
        locGrad = error*dedy*dydz
        outputLocalGrads[x] = locGrad
    return outputLocalGrads

#find weight grads for batch example
def WeightGrads(s, grads, inputs, localGrads):
    for y in range(0, inputs[s].shape[0]): #for each neuron in previous layer
        for o in range(0, localGrads[s].shape[0]): #for each neuron in next layer
            grads[o,y] += inputs[s][y]*localGrads[s][o]

#find the weights for a layer for a batch
def WeightGradsBatch(batch, weights, inputs, localGrads):
    grads = np.zeros(weights.shape)
    for s in range(0, len(batch)):
        WeightGrads(s, grads, inputs, localGrads)
    grads = -lr*grads
    grads = grads/len(batch) #mini batch averaging
    return grads

#find partial derivs for a mini batch example
def PartialDerivs(s, inputs, weights, localGrads):
    partialDerivs = np.zeros(inputs[s].shape[0])
    for z in range(0, inputs[s].shape[0]): #for each neuron in previous layer
        for o in range(0, localGrads[s].shape[0]): #for each neuron in next layer
            partialDerivs[z] += weights[o,z]*localGrads[s][o]
        partialDerivs[z] = partialDerivs[z]*sigmoidDeriv(inputs[s][z])
    return partialDerivs

#find partial derivs for a batch
def PartialDerivsBatch(batch, inputs, weights, localGrads):
    partialDerivsBatch = np.ndarray(inputs.shape)
    for s in range(0, len(batch)):
        partialDerivsBatch[s] = PartialDerivs(s, inputs, weights, localGrads)
    return partialDerivsBatch

#endregion

#params
trainingSet = trainingSin.copy()
np.random.shuffle(trainingSet)
lr = 6e-3
batch_size = trainingSet.shape[0] if trainingSet.shape[0] < 32 else 32 
batchUpdates = 20000000

#TRAINING STOCHASITC GRADIENT
t = 0
for j in range(0, batchUpdates):

    #generate batch
    batch = []
    if t+batch_size <= trainingSet.shape[0]: #room for full batch
        batch = trainingSet[t:t+batch_size]
        t += batch_size
    elif t == trainingSet.shape[0]: #reset case #1
        np.random.shuffle(trainingSet)
        t = 0
        batch = trainingSet[t:t+batch_size]
    else: #reset case #2
        batch = trainingSet[t:]
        np.random.shuffle(trainingSet)
        t = 0

    #FEEDFORWARD and store
    inputs, z1s, y1s, z2s, y2s, z3s, y3s, z4s, outputs, errors = np.empty((0,2)), np.empty((0,20)), np.empty((0,21)), np.empty((0,20)), np.empty((0,21)), np.empty((0,20)), np.empty((0,21)), np.empty((0,1)), np.empty((0,1)), np.empty((0,1))
    for x in range(0, len(batch)):
        el = batch[x]
        input = el[0]
        label = el[1]
        out = inference(input, label)
        inputs = np.append(inputs, np.array([out[0]]), 0)
        z1s = np.append(z1s, np.array([out[1]]), 0)
        y1s = np.append(y1s, np.array([out[2]]), 0)
        z2s = np.append(z2s, np.array([out[3]]), 0)
        y2s = np.append(y2s, np.array([out[4]]), 0)
        z3s = np.append(z3s, np.array([out[5]]), 0)
        y3s = np.append(y3s, np.array([out[6]]), 0)
        z4s = np.append(z4s, np.array([out[7]]), 0)
        outputs = np.append(outputs, np.array([out[8]]), 0)
        errors = np.append(errors, np.array([out[9]]), 0)

    #BACKPROP
    outputLocalGradsBatch = OutputLocalGradsBatch(batch, z4s, outputs, errors)

    w4Grads = WeightGradsBatch(batch, w4, y3s, outputLocalGradsBatch)

    z3PartialDerivs = PartialDerivsBatch(batch, z3s, w4, outputLocalGradsBatch)

    w3Grads = WeightGradsBatch(batch, w3, y2s, z3PartialDerivs)

    z2PartialDerivs = PartialDerivsBatch(batch, z2s, w3, z3PartialDerivs)

    w2Grads = WeightGradsBatch(batch, w2, y1s, z2PartialDerivs)
    
    z1PartialDerivs = PartialDerivsBatch(batch, z1s, w2, z2PartialDerivs)

    w1Grads = WeightGradsBatch(batch, w1, inputs, z1PartialDerivs)

    #UPDATE LAYERS
    w4 += w4Grads
    w3 += w3Grads
    w2 += w2Grads
    w1 += w1Grads

    #LOGGING
    if j % 100 == 0:
        print(inputs[0], outputs[0], np.sum(errors))