import numpy as np


def relu(x):
   return np.maximum(0,x)

def d_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return x*(1-x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,3)
        # ignoring biases
        self.y = y
        self.output = np.zeros(y.shape)

    def forwardprop(self):
        ### ignoring biases
        self.layer1 = relu(np.dot(self.input, self.weights1))
        self.layer2 = relu(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        #get derivative of the loss function (sum of squares loss function) wrt weights 1 and 2
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)*d_relu(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * d_relu(self.output), self.weights2.T) * d_relu(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, x, y):
        self.output = self.forwardprop()
        self.backprop()




