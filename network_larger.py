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
    def __init__(self, x, y, hl1, hl2):
        """
        hl1 - number of nodes in first hidden layer
        hl2 - number of nodes in second hidden layer
        out - number of output nodes
        """
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],hl1)
        self.weights2 = np.random.rand(hl1,hl2)
        self.weights3 = np.random.rand(hl2,y.shape[1])
        # ignoring biases
        self.y = y
        self.output = np.zeros(y.shape)

    def forwardprop(self):
        ### ignoring biases
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer3

    def backprop(self):
        #get derivative of the loss function (sum of squares loss function) wrt weights 1 and 2
        d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output)*d_sigmoid(self.output)))
        d_weights2 = np.dot(self.layer1.T,  (np.dot(2*(self.y - self.output) * d_sigmoid(self.output), self.weights3.T) * d_sigmoid(self.layer2)))
        d_weights1 = np.dot(self.input.T,  np.dot((np.dot(2*(self.y - self.output) * d_sigmoid(self.output), self.weights3.T) * d_sigmoid(self.layer2)), self.weights2.T)*d_sigmoid(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def train(self):
        self.output = self.forwardprop()
        self.backprop()

    def test(self, x):
        self.layer1 = sigmoid(np.dot(x, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer3

