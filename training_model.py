'''training the model'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = 1500))
classifier.add(Dropout(0.6))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.6))

classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.6))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)