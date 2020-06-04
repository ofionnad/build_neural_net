import numpy as np
import network_larger as nw
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

#some v basic training data
x = np.array(([0,0,1], [0,1,1], [1,0,1], [1,1,1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)

# lets try the iris dataset
iris = datasets.load_iris()
X = iris.data[:100, :2]
X = normalize(X, axis=0)
Y = iris.target[:100]
Y = np.array([[x] for x in Y])
Y_c = np.zeros((len(Y), 2))
for i,j in enumerate(Y):
    if j==0:
        Y_c[i] = [1, 0]
    elif j==1:
        Y_c[i] = [0, 1]
    elif j==2:
        pass

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_c, test_size=0.3, random_state=11)

nn = nw.NeuralNetwork(X_train, Y_train, 4, 4)

#train it
loss_trend = []
for i in range(5000):
    if i%100==0:
        loss = np.mean(np.square(Y_train - nn.forwardprop()))
        loss_trend.append(loss)
        print("\n")
        print("iteration: {}\n".format(i))
        print("predicted output \n {}".format(nn.forwardprop()))
        print("Loss: \n {}\n".format(loss))

    nn.train()
