############################
##
## Using keras with sklearn datasets to show how easy it is to use
##
###########################
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


## import some datasets
wine = datasets.load_wine()

### Lets try training a neural network to predict flower type in iris dataset
df = pd.DataFrame(normalize(wine.data, axis=0)) #include normalisation of data
Y_data = wine.target

# here I will try splitting the target data categorising it into arrays
Y_data = np_utils.to_categorical(Y_data)

# build training and testing set
X_train, X_test, y_train, y_test = train_test_split(df.values, Y_data, test_size=0.2, random_state=44)

#build a model
model = Sequential()
model.add(Dense(52, input_shape=(13,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(13, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

## Note that if you want to monitor your metrics, tensorboard in recommended!!

print(model.summary())

training = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size=10, epochs=500)

results = model.evaluate(X_test, y_test)
print("Loss:\t{}\nAccuracy:\t{}\n".format(results[0], results[2]))

#precision = precision_score(y_test, model.predict_classes(X_test)[:,0], average='weighted')
#recall = recall_score(y_test, model.predict_classes(X_test)[:,0], average='weighted')
#f1 = f1_score(y_test, model.predict_classes(X_test)[:,0], average='weighted')
#print("Precision:\t{}\n".format(precision))

plt.subplot(211)
plt.plot(training.history['loss'], label='train')
plt.plot(training.history['val_loss'], label='test')
plt.title('Loss')
plt.legend()
plt.subplot(212)
plt.plot(training.history['accuracy'], label='train')
plt.plot(training.history['val_accuracy'], label='test')
plt.title('Accuracy')
plt.legend()
plt.show()

