'''
NOTES:
 * Possible employ simulated annealing by decreasing mutation deviance
   parameters for chromosomes over time.
DOIT:
 * If a Chromosome is initialized as a copy of another, it should not require
   parameters for the number of input and output neurons.
'''

import ANN
import math
from random import random, randint, seed as srand
import sys
from time import time

output = sys.stdout
srand(time())

def setOutput(out):
    global output
    output = out
    ANN.setOutput(out)

class Chromosome():
    """
    Unit of a population. Contains parameters for an ANN structure.
    """
    initLayers = (0, 3)
    initLearnRate = (0.0, 1.0)
    initTrainEpochs = (50, 30000)
    learn_step = 0.01
    def __init__(self, inNrns, outNrns, mirror=None):
        """
        @param inNrns: Number of input neurons for ANN.
        @param outNrns: Number of output neurons for ANN.
        @param mirror (optional): Chromosome object. New chromosome will
          inherit all attributes of this chromosome.
        """
        if mirror != None:
            self.copy(mirror)
            return
        self.layers = [inNrns, outNrns]
        for l in range(randint(*Chromosome.initLayers)):
            self.addRandLayer()
        self.learnRate = randint(
            self.initLearnRate[0]/self.learn_step,
            self.initLearnRate[1]/self.learn_step ) * self.learn_step
        self.trainEpochs = randint(*self.initTrainEpochs)
        self.sqrErr = 1000
        self.fitness = 0.0

    def copy(self, mirror):
        """
        Overwrites own properties with that of another chromosome.
        @param mirror: chromosome object to mirror.
        """
        self.layers = mirror.layers
        self.learnRate = mirror.learnRate
        self.trainEpochs = mirror.trainEpochs
        self.sqrErr = mirror.sqrErr
        self.fitness = mirror.fitness

    initNeurons = (1, 8)
    def addRandLayer(self):
        """
        Adds a new hidden layer to chromosome's ANN.
        Position of the new layer is randomly determines.
        Quantity of neurons in the new layer is randomly determined.
        """
        # Don't manipulate the input and output layers
        l_i = randint(1, len(self.layers)-1)
        self.layers.insert(l_i, randint(*self.initNeurons))

    def rmvRandLayer(self):
        """
        Deletes a random hidden layer from chromosome's ANN
        """
        if len(self.layers) < 3:
            return
        l_i = randint(1, len(self.layers)-2)
        self.layers.pop(l_i)

    # Amount a parameter can shift in either direction
    mLearn = 1.0
    mLayers = 2
    mNeurons = 5
    mTrainEpochs = 20000
    minTrainEpochs = 50
    def mutate(self):
        """
        Mutates a chromosome
        """
        choice = randint(1, 3)
        if choice == 1:
            # Mutate learning rate.
            low = (
                -self.mLearn if self.learnRate > self.mLearn
                else -self.learnRate + self.learn_step
            )
            change = randint(
                int(low/self.learn_step),
                int(self.mLearn/self.learn_step - 1)
            )
            # Cannot add/subtract 0.
            if change >= 0: change += 1
            change *= self.learn_step
            self.learnRate += change

        elif choice == 2:
            # Mutate layers
            if len(self.layers) < 3 or randint(1, 2) == 1:
                # Mutate number of layers
                low = (
                    -self.mLayers if len(self.layers)-2 >= self.mLayers
                    else -(len(self.layers)-2)
                )
                change = randint(low, self.mLayers-1)
                # Cannot add/subtract 0 layers.
                if change >= 0: change += 1
                if change > 0:
                    for l in range(change):
                        self.addRandLayer()
                if change < 0:
                    for l in range(change):
                        self.rmvRandLayer()

            else:
                # Mutate number of neurons in a layer
                # Pick a random layer to mutate.
                # Assumes there is at least one hidden layer.
                # Don't touch input and output layers
                l_i = randint(1, len(self.layers)-2)
                # Must be at least 1 neuron left after mutation
                low = (
                    -self.mNeurons if self.layers[l_i] > self.mNeurons
                    else -(self.layers[l_i]-1)
                )
                # Amount of neurons to delete or add.
                change = randint(low, self.mNeurons-1)
                # Cannot add/subtract 0 neurons.
                if change >= 0: change += 1
                self.layers[l_i] += change

        elif choice == 3:
            # Mutate number of training epochs
            self.trainEpochs += randint(-self.trainEpochs, self.trainEpochs)
            if self.trainEpochs < self.minTrainEpochs:
                self.trainEpochs = self.minTrainEpochs

    # mutation probability
    mProb = 0.3
    def tryMutate(self):
        """
        Has chromosome mutate if a predefined probability is met.
        """
        if random() > self.mProb:
            self.mutate()

    # All lambdas are transformations of a logistic function
    # mirrored across the y-axis
    # Modify the dividend to change the impact of a fitness lambda.
    _fit_sqrErr  = lambda self,x: 5 / (1 + math.exp(x*8 - 4))
    _fit_layers  = lambda self,x: 3 / (1 + math.exp(x*1.6 - 6))
    _fit_hidNrns = lambda self,x: 2 / (1 + math.exp(x*0.8 - 14))
    _fit_trainEpochs = lambda self,x: 1 / (1 + math.exp(x*0.00008 - 5))
    _fitPow = 10
    maxFit = (_fit_sqrErr(None, 0) + _fit_layers(None, 2) +
              _fit_hidNrns(None, 0) + _fit_trainEpochs(None, minTrainEpochs)
    ) ** _fitPow
    def update(self, testData):
        """
        Updates chromosome's ANN based on changes made to layers.
        @param testData: Used to train ANN and assess its fitness.
        Total length of list/tuple should be not more and no less than the sum
        of the chromosome's number of input and output neurons. First indices
        will be input(s). Last indices will be expected output(s).
        """
        testANN = ANN.ANN(self.layers, self.learnRate)
        # Train ANN.
        i = 0
        for e in range(self.trainEpochs):
            testANN.feedforward(testData[i][:self.layers[0]])
            testANN.backpropagate(testData[i][-self.layers[-1]:])
            i += 1
            if i >= len(testData): i = 0

        # Find average squared error.
        self.sqrErr = 0
        for row in testData:
            testANN.feedforward(row[:self.layers[0]])
            for n in range(0, self.layers[-1]):
                self.sqrErr += (row[self.layers[0]+n] - testANN.out[n]) ** 2
        self.sqrErr /= self.layers[-1]

        totalNrns = 0
        for l in range(1, len(self.layers)-1):
            totalNrns += self.layers[l]

        # The lower these numbers are, the higher the fitness value.
        self.fitness = self._fit_sqrErr(self.sqrErr)
        self.fitness += self._fit_layers(len(self.layers))
        self.fitness += self._fit_hidNrns(totalNrns)
        self.fitness **= self._fitPow

class Population:
    """
    Contains a population of chromosomes.
    """
    def __init__(self, inNrns, outNrns, testData, maxChroms):
        """
        @param inNrns: Number of input neurons for each chromosome's ANN.
        @param outNrns: Number of output neurons for each chromosome's ANN.
        @param testData: Used to train a chromosome's ANN and assess its
         fitness. Each index should contain a list, which corresponds to a
         test iteration. The first values in this list correspond to the input
         values for an ANN. The last values in the table correspond to the
         desired output data from such input. Likewise, the length of each list
         should correspond. Likewise, the length of each test iteration list
         should be the sum of inNrns + outNrns. The number of test
         iterations/indices can be of arbitrary length, but must have at least
         one index.
        @param maxChroms: Maximum number of chromosomes in population.
        """
        self.chroms = [Chromosome(inNrns, outNrns) for i in range(maxChroms)]
        self.testData = testData

    def fitCheck(self):
        """
        Performs a fitness check on population and updates all of its
        chromosomes' fitness values.
        """
        self.avgSqrErr = 0.0
        self.totalFit = 0.0
        self.best_sqrErr = self.chroms[0]
        self.best_fit = self.chroms[0]
        c_i = 1
        for c in self.chroms:
            c_i += 1
            c.update(self.testData)
            self.avgSqrErr += c.sqrErr
            if c.sqrErr < self.best_sqrErr.sqrErr:
                self.best_sqrErr = c
            if c.fitness > self.best_fit.fitness:
                self.best_fit = c
            self.totalFit += c.fitness
        self.avgSqrErr /= len(self.chroms)
        self.avgFit = self.totalFit / len(self.chroms)

    def selectPar(self):
        """
        Returns a randomly selected chromosome from population. Chromosomes
        with higher fitness values have a higher chance of getting chosen.
        @return: Chromosome object.
        """
        fitMark = random() * self.totalFit
        sumFit = 0
        c = None
        for c in self.chroms:
            sumFit += c.fitness
            if sumFit >= fitMark:
                break
        return c

    def reproduce(self, parPop):
        """
        Reproduce with another population. Or rather, bear the offsprings of
        another population. Population stored in self is overwritten.
        @param parPop: Population object. Population to bear offspring of.
        """
        for c in self.chroms:
            c.copy(parPop.selectPar())
            c.tryMutate()

class Simulation():
    """
    Contains multiple populations and performs genetic algorithms on them.
    """
    def __init__(self, inNrns, outNrns, testData, maxChroms):
        """
        @param inNrns: Number of input neurons for each chromosome's ANN.
        @param outNrns: Number of output neurons for each chromosome's ANN.
        @param testData: Used to train a chromosome's ANN and assess its
         fitness. See population.__init__().
        @param maxChroms: Maximum number of chromosome
        """
        self.pops = [Population(inNrns, outNrns, testData, maxChroms) for i in range(2)]
        self.curPop = 0
        self.best_sqrErr = {
            'chrom': Chromosome(-1, -1, mirror=self.pops[self.curPop].chroms[0]),
            'gen': -1
        }
        self.best_fit = {
            'chrom': Chromosome(-1, -1, mirror=self.pops[self.curPop].chroms[0]),
            'gen': -1
        }
        self.gen = 0

    def nextPop(self):
        """
        Returns the index of the next population, the population other than
        the current population.
        """
        return (1 if self.curPop == 0 else 0)

    def update(self):
        """
        Performs selection, reproduction, and fitness checking.
        """
        output.write("generation: {}\n".format(self.gen))
        thisPop = self.pops[self.curPop]
        thisPop.fitCheck()
        if thisPop.best_sqrErr.sqrErr < self.best_sqrErr['chrom'].sqrErr:
            self.best_sqrErr['chrom'].copy(thisPop.best_sqrErr)
            self.best_sqrErr['gen'] = self.gen
        if thisPop.best_fit.fitness > self.best_fit['chrom'].fitness:
            self.best_fit['chrom'].copy(thisPop.best_fit)
            self.best_fit['gen'] = self.gen
        self.curPop = self.nextPop()
        nextPop = self.pops[self.curPop]
        nextPop.reproduce(thisPop)
        self.gen += 1

    def outputPopInfo(self, pop):
        output.write("    This time:\n")
        output.write("\tAverage Error Squared: {}\n".format(pop.avgSqrErr))
        output.write("\tAverage Fitness:       {}\n".format(pop.avgFit))
        output.write("\tTotal Fitness:         {}\n".format(pop.totalFit))
        output.write("\tLowest Squared Error:  {}\n".format(
            pop.best_sqrErr.sqrErr) )
        output.write("\tMax Fitness:           {}\n".format(
            pop.best_fit.fitness) )

        output.write("    All-time:\n")
        output.write("\tLowest Squared Error:  {}\n".format(
            self.best_sqrErr['chrom'].sqrErr) )
        output.write("\tMax Fitness:           {}\n".format(
            self.best_fit['chrom'].fitness) )
        output.flush()

    def outputChromInfo(self, chrom):
        """
        self.layers = [inNrns, outNrns]
        for l in range(randint(*Chromosome.initLayers)):
            self.addRandLayer()
        self.learnRate = randint(
            self.initLearnRate[0]/self.learn_step,
            self.initLearnRate[1]/self.learn_step ) * self.learn_step
        self.trainEpochs = randint(*self.initTrainEpochs)
        self.sqrErr = 1000
        self.fitness = 0.0
        """
        output.write("\tNeurons-Per-Layers:\n")
        totalNrns = 0
        for i, nrns in enumerate(chrom.layers):
            output.write("\t    #{} -> {}\n".format(i, nrns))
            totalNrns += nrns
        output.write("\tLayer Count:     {}\n".format(len(chrom.layers)))
        output.write("\tNeuron Count:    {}\n".format(totalNrns))
        output.write("\tTraining Epochs: {}\n".format(chrom.trainEpochs))
        output.write("\tLearning Rate:   {}\n".format(chrom.learnRate))
        output.write("\tSquared Error:   {}\n".format(chrom.sqrErr))
        output.write("\tFitness:         {}\n".format(chrom.fitness))

    def simulate(self, gens):
        """
        Perform the genetic algorithm. Call self.update() in a loop.
        """
        for g in range(gens):
            thisPop = self.pops[self.curPop]
            self.update()
            self.outputPopInfo(thisPop)
            # Check for convergance.
            if thisPop.avgFit / thisPop.best_fit.fitness > 0.98:
                output.write("Chromosomes have converged. Ending simulation.\n")
                break

        output.write("\nAGGREGATE RESULTS:\n")
        output.write("    Lowest Squared Error:\n")
        output.write("\tGeneration:      {}\n".format(self.best_sqrErr['gen']))
        self.outputChromInfo(self.best_sqrErr['chrom'])
        output.write("    Max Fitness:\n")
        output.write("\tGeneration:      {}\n".format(self.best_fit['gen']))
        self.outputChromInfo(self.best_fit['chrom'])
        output.flush()
