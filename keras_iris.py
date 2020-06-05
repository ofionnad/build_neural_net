############################
##
## Using keras with sklearn datasets to show how easy it is to use
##
###########################
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import normalize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


## import some datasets
iris = datasets.load_iris()

### Lets try training a neural network to predict flower type in iris dataset
df = pd.DataFrame(normalize(iris.data, axis=0)) #include normalisation of data
df['target'] = iris.target

# build training and testing set
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:4], df['target'], test_size=0.2, random_state=12)

#build a model
model = Sequential()
model.add(Dense(16, input_shape=(4,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='sgd', metrics=['mean_squared_error', 'accuracy'])

print(model.summary())

training = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size=10, epochs=200)

results = model.evaluate(X_test, y_test)
print("Loss:\t{}\nAccuracy:\t{}\n".format(results[0], results[1]))

precision = precision_score(y_test, model.predict_classes(X_test)[:,0], average='weighted')
recall = recall_score(y_test, model.predict_classes(X_test)[:,0], average='weighted')
f1 = f1_score(y_test, model.predict_classes(X_test)[:,0], average='weighted')
print("Precision:\t{}\n".format(precision))

plt.subplot(211)
plt.plot(training.history['loss'], label='train')
plt.plot(training.history['val_loss'], label='test')
plt.legend()
plt.subplot(212)
plt.plot(training.history['accuracy'], label='train')
plt.plot(training.history['val_accuracy'], label='test')
plt.legend()
plt.show()

