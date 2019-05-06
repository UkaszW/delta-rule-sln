import random
import sys
import os
import numpy
import time

TRAINING_STEP = 0.005
EPOCHS = 10000
GRADIENT = 0.000001

file = open('../example-patterns/patterns4.txt', 'r')
line1 = (file.readline())
line2 = (file.readline())

nr_of_patterns = int(line1[12:14])
nr_of_xcomp = 10


X = numpy.zeros((nr_of_patterns, nr_of_xcomp))
Z = numpy.zeros(nr_of_patterns)
Z = Z.reshape((nr_of_patterns, 1))
sumOfDelta = numpy.ones(nr_of_patterns)
for i in range(0,nr_of_patterns):
    line = file.readline()
    for j in range(0, nr_of_xcomp):
        X[i,j] = float(line[j*10:j*10+9])

    Z[i] = float(line[j*10+9:])



print('X: \n', X)
print('Z: \n', Z)


class Neuron:
    weightsVector = None

    def __init__(self, numOfFeatures):  # za numOfFeatures podstawiamy ilosc wejsc

        self.weightsVector = numpy.random.rand(numOfFeatures, 1)

    def show(self):
        print('\n Final weights: \n', self.weightsVector)


neuron2 = Neuron(X.shape[1])  # ilosc ficzerów jest równa ilosci kolumn macierzy X



class Training:

    inputMatrix = None
    outputVectorZ = None
    __epochs = EPOCHS
    __trainingStep = 0.0001
    neuron0 = None

    def __init__(self, X, Z, neuron, step):
        self.inputMatrix = X
        self.outputVectorZ = Z
        self.neuron0 = neuron
        self.trainingStep = step


    def singleTraining(self):

        Gradient = GRADIENT
        inc = 0
        delta = 1000

        print('\n Training step:',self.trainingStep)
        print('\n Epochs:',self.__epochs)
        print(' Maximal accepted cost value:', Gradient)

        while numpy.sum(sumOfDelta)>Gradient and inc < self.__epochs:

            # brakuje tasowania rzedów macierzy X

              #mnozenie macierzy. Docelowo ma wyjsc Y=[ilosc prób,1]

            o = 0
            for k in range(0,self.inputMatrix.shape[0]):
                Y = numpy.matmul(self.inputMatrix[k, :], self.neuron0.weightsVector)

                delta = self.outputVectorZ[k] - Y
                sumOfDelta[o]= (delta*delta)
                if o<nr_of_patterns:
                    o+=1
                else:
                    o=0

                #print('Iteration:',inc,'\nDelta \n', k, delta,'\n')
                for i in range(0,self.inputMatrix.shape[1]):
                    self.neuron0.weightsVector[i] = self.neuron0.weightsVector[i] + self.trainingStep * delta * self.inputMatrix[k, i]

            inc += 1
        #print('Delta \n', delta)
        print('\n Number of iterations:', inc)
        print(' Sum of squared deltas:', numpy.sum(sumOfDelta))

        return self.neuron0


start = time.time()

training1 = Training(X, Z, neuron2, TRAINING_STEP)
neuron2 = training1.singleTraining()

czas = time.time() - start

neuron2.show()
print('\n Time of calculations: \n', czas*1000,'ms')