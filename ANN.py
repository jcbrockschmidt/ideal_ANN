import math
from   random import seed as srand, random as rand
import sys
from   time import time

output = sys.stdout
srand(time())

def sigmoid(x):
    return (1.0 / (1.0 + math.exp(-x)))

def sigmoidDeriv(x):
    return (x * (1.0 - x))

class ANN:
    def __init__(self, layers, learnRate, weights=[]):
        self.inp = [ None for i in range(layers[0]) ]
        self.hid = [ [None for i in range(layers[l])] for l in range(1, len(layers)-1) ]
        self.out = [ None for i in range(layers[-1]) ]

        self.learnRate = learnRate

        # If weights are specified, make sure given table is valid.
        weightsDeclared = True
        if len(weights) > 0:
            if len(weights) == len(layers):
                for lyr in range(len(layers)):
                    if len(weights[lyr]) != layers[lyr]:
                        output.write("ERROR: invalid weights argument for ANN; there are " +
                              str(layers[lyr]) +
                              "neurons in layer " +
                              str(len(layers)) +
                              "layers; weights will be randomized\n")
                        weightsDeclared = False
                        break

                self.weights = weights[:]
            else:
                output.write("ERROR: invalid weights argument for ANN; there are " +
                      str(len(layers)) +
                      "layers; weights will be randomized\n")
                weightsDeclared = False
        else:
            weightsDeclared = False

        # If weights self.weights have not been declared
        if not weightsDeclared:
            self.weights = []
            for lyr in range(len(layers)-1):
                self.weights.append([])
                for n in range(layers[lyr]):
                    # Add additional weight (+1) for bias.
                    self.weights[lyr].append(
                        [ ANN._randWeight() for n2 in range(layers[lyr+1]+1) ]
                    )

    def _randWeight():
        return rand() - 0.5

    def feedforward(self, ins):
        # Validate arguments
        if len(ins) != len(self.inp):
            output.write("ERROR: could not feed forward ANN; there are " +
                  str(len(self.inp)) +
                  " input neurons\n")
            return

        # Update input layer
        self.inp = ins[:]

        # Calculate for hidden layers
        lastLyr = self.inp
        for hid_i in range(len(self.hid)):
            # Calculate for individual layer's nuerons
            for n_cur in range(len(self.hid[hid_i])):
                sum = 0.0
                for n_last in range(len(lastLyr)):
                    # Note that index of hidden node in self.weights is
                    # 1 greater than its index in self.hid and hid_err
                    # Thus, in self.weights, hid_i references
                    # the layer before it
                    sum += (
                        lastLyr[n_last] * self.weights[hid_i][n_last][n_cur]
                    )

                    # Add bias
                    sum += self.weights[hid_i][-1][n_cur]

                    self.hid[hid_i][n_cur] = sigmoid(sum)

            lastLyr = self.hid[hid_i]

        # Calculate for output layer
        for n_cur in range(len(self.out)):
            sum = 0.0
            for n_last in range(len(lastLyr)):
                sum += lastLyr[n_last] * self.weights[-1][n_last][n_cur]

            # Add bias
            sum += self.weights[-1][-1][n_cur]

            self.out[n_cur] = sigmoid(sum)

        # Feedforwarding was successful
        return True

    def backpropagate(self, target):
        # Validate arguments
        if len(target) != len(self.out):
            output.write("ERROR: could not backpropagate ANN; there are " +
                  str(len(self.inp)) +
                  " input neurons\n")
            return False

        # Calculate output layer error
        out_err = []
        for n in range(len(self.out)):
            out_err.append((target[n] - self.out[n]) * sigmoidDeriv(self.out[n]))

        # Calculate hidden layers' error
        hid_err = [ [] for h in range(len(self.hid))]
        lastLyr = self.out
        lastErr = out_err
        # From top to bottom; modify weights that lead into hidden node
        for hid_i in range(len(self.hid)-1, -1, -1):
            for n_cur in range(len(self.hid[hid_i])):
                hid_err[hid_i].append(0.0)
                for n_last in range(len(lastErr)):
                    # Note that index of hidden node in self.weights is
                    # 1 greater than its index in self.hid and hid_err
                    # Thus, in self.weights, hid_i references
                    # the layer before it
                    hid_err[hid_i][n_cur] += lastErr[n_last] * self.weights[hid_i+1][n_cur][n_last]

                hid_err[hid_i][n_cur] *= sigmoidDeriv(self.hid[hid_i][n_cur])

            lastLyr = self.hid[hid_i]
            lastErr = hid_err[hid_i]

        # Calculate new weights and biases for hidden nodes
        lastLyr = self.inp
        # From bottom to top
        for hid_i in range(len(self.hid)):
            for n_cur in range(len(self.hid[hid_i])):
                for n_last in range(len(lastLyr)):
                    # Note that index of hidden node in self.weights is
                    # 1 greater than its index in self.hid and hid_err
                    # Thus, in self.weights, hid_i references
                    # the layer before it
                    self.weights[hid_i][n_last][n_cur] += (
                        self.learnRate * hid_err[hid_i][n_cur] * lastLyr[n_last]
                    )

                # Update the bias
                self.weights[hid_i][-1][n_cur] += (
                    self.learnRate * hid_err[hid_i][n_cur]
                )

            lastLyr = self.hid[hid_i]

        # Calculate new weights and biases for output nodes
        for n_cur in range(len(self.out)):
            for n_last in range(len(lastLyr)):
                self.weights[-1][n_last][n_cur] += (
                    self.learnRate * out_err[n_cur] * lastLyr[n_last]
                )

            # Update the bias
            self.weights[-1][-1][n_cur] += (
                self.learnRate * out_err[n_cur]
            )

        # Backpropagation was successful
        return True

    def errSqr(self, target):
        # Validate arguments
        if len(target) != len(self.out):
            output.write("ERROR: could not get error squared for ANN; there are " +
                  str(len(self.inp)) +
                  " input neurons")
            return False

        err = 0.0
        for n_out in range(len(self.out)):
            err += (self.out[n_out] - target[n_out])**2
        err *= 0.5

        return err
