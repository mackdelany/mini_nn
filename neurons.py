import numpy as np

class Perceptron():

    def __init__(self, input_size, bias=0):
        self.weights = np.random.rand(input_size, 1)
        self.bias = bias

    def forward(self, input_vector):
        input_vector = np.array(input_vector).reshape(-1, 1)
        return sum(self.weights * input_vector) + self.bias

class BinaryNeuron(Perceptron):

    def activate(self, input_vector):
        x = self.forward(input_vector)
        return 0 if x < 0 else 1

class TanHNeuron(Perceptron):

    def activate(self, input_vector):
        x = self.forward(input_vector)
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

class ReLuNeuron(Perceptron):

    def activate(self, input_vector):
        x = self.forward(input_vector)
        return 0 if x <= 0 else x
