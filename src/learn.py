from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import scipy
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt


def knn_fit(trainX, trainy):
  clf = neighbors.KNeighborsClassifier(5, 'distance')
  clf.fit(trainX, trainy)
  return clf

def knn_test(clf, testX, testy):
  predictions = clf.predict(testX)
  c = confusion_matrix(testy, predictions)
  return c

def nn_fit(trainX, trainy):
  traincaty = np_utils.to_categorical(trainy)
  nout = traincaty.shape[1]
  print("NOUT: ", nout)

  model = Sequential()
  model.add(Dense(64, input_dim=62, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(nout, activation='softmax'))

  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
  model.fit(trainX, traincaty, epochs=40, batch_size=10)
  return model

def nn_test(model, testX, testy=None):
  predictions = model.predict(testX)
  predictions = predictions.argmax(axis=1)
  if testy is not None:
    scores = model.evaluate(testX, np_utils.to_categorical(testy))
    c = confusion_matrix(testy, predictions)
    return c
  else:
    return predictions

def fit_validate(X, y, fitfunc=knn_fit, testfunc=knn_test, display=False):
  Xy = np.hstack([X.T, y.T])
  np.random.shuffle(Xy)
  samples = int(Xy.shape[0]*0.75)
  trainX, trainy = Xy[:samples,:-1], Xy[:samples,-1]
  testX, testy = Xy[samples:,:-1], Xy[samples:,-1]

  model = fitfunc(trainX, trainy)
  c = testfunc(model, testX, testy)
  if display:
    plt.imshow(c)
    plt.show()
  return c, model

