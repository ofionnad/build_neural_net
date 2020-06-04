# build_neural_net
Writing a basic neural network for better understanding. 

Uses the iris dataset from sklearn. 

Ignored the biases for the purposes of simplicity. 

Neural network is composed of inputs (data you want to fit) and weights and biases, resulting in an output value. 
The neural network is 'trained' by feedforward propogation, and backpropagation to update the weights of the connecting nodes. 

### Training
For the purposes of simplicity the network was trained on a small set of data of 1's and 0's and compared to an output array.

Later I tried train a new larger network using the iris dataset, using the first two parameters, and 2 hidden layers. 

The network does a bad job at prediction, and suffers from many problems. This is why people do not write their own Neural networks. But a great way to learn the workings of a NN. 

Here is the loss function when trained on the iris data set for a 2x4x4x2 NN:
![Loss Trend](https://github.com/ofionnad/build_neural_net/blob/master/loss_vs_iteration.png "Loss function vs iteration")

For small scale networks this can be coded up manually. 

For anything else tools like keras, tensorflow, pytorch are highly recommended. 
