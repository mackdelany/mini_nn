import numpy as np

class Perceptron():

    def __init__(self, input_size, bias=0):
        self.weights = np.random.rand(input_size, 1)
        self.bias = bias

    def forward(input_vector):
        return np.dot(self.weights, input_vector) + self.bias


class ReLuNeuron(Perceptron):

    def activate(x):
        if x <= 0:
            return 0
        else :
            return x


class BinaryNeuron(Perceptron):

    def activate(x):
        if x < 0:
            return 0
        else:
            return 1


class TanhNeuron(Perceptron):

    def activate(x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
