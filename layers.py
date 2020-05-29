from neurons import BinaryNeuron, TanHNeuron, ReLuNeuron

class Layer():

    def forward(self, input_vector):
        return [neuron.activate(input_vector) for neuron in self.neurons]

class BinaryLayer(Layer):

    def __init__(self, inputs, outputs):
        self.neurons = [BinaryNeuron(inputs) for _ in range(outputs)]

class TanHLayer(Layer):

    def __init__(self, inputs, outputs):
        self.neurons = [TanHNeuron(inputs) for _ in range(outputs)]

class ReLuLayer(Layer):

    def __init__(self, inputs, outputs):
        self.neurons = [ReLuNeuron(inputs) for _ in range(outputs)]