#!/usr/bin/python3

from ANN import ANN
from random import seed as srand, randint
from time import time

srand(time())

# Test data for a XOR gate
testData = [
    [0.1, 0.1, 0.9],
    [0.1, 0.9, 0.9],
    [0.9, 0.1, 0.9],
    [0.9, 0.9, 0.1]
]

# Create ANN with 2 input neurons, 1 hidden layer with 3 neurons,
# 1 output neuron, and a learning rate of 10.0
net = ANN([2, 3, 1], 3.0)

# Train network
for i in range(10000):
    #testRow = testData[i % len(testData)]
    testRow = testData[randint(0, len(testData)-1)]
    net.feedforward(testRow[:-1])

    # Calculate and display error squared
    print("err: " + str(net.errSqr(testRow[-1:])))

    net.backpropagate(testRow[-1:])

accuracy = 0.0
for testRow in testData:
    net.feedforward(testRow[:-1])
    accuracy += net.errSqr(testRow[-1:])

    matching = (
        (testRow[-1] >= 0.45 and net.out[-1] >= 0.45) or
        (testRow[-1] < 0.45 and net.out[-1] < 0.45)
    )

    print(str(testRow[0]) +
          "\t" + str(testRow[1]) +
          "\t:\t" +
          str(net.out[0]) + 
          "\t" +
          ("GOOD" if matching else "BAD")
    )

accuracy /= len(testData)
print("Aggregate accuracy: " + str(accuracy))
