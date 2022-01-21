import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
import sys 

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#python3 forecast.py -d nasd_quer.csv -n 5
file = sys.argv[1]
# file = "/content/drive/MyDrive/nasdaq2007_17.csv"
# df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/nasdaq2007_17.csv", sep="\t")

# file = "/content/nasd_query.csv"

df=pd.read_csv(file,sep="\t")
shape = df.shape
print("Number of rows and columns:",shape)
colls=shape[1]
rows=shape[0]
n=rows
window=60
print("Number of rows:",rows)
print("Number of colls:",colls)

trainsize=((80*colls)//100)
testsize=((20*colls)//100)

print("trainsize:",trainsize)
print("testsize:",testsize)
model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (window, 1)))
model.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))# Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

train = df.iloc[:,1:trainsize].values
qt = QuantileTransformer()
training_set_scaled=qt.fit_transform(train.reshape(-1,1))
# Compiling the RNN
lag =1
for j in range(n):
    name=df.iloc[j,0:1].values

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(window, trainsize-1):
        X_train.append(training_set_scaled[(j+1)*i-window:(1+j)*i,0])
        y_train.append(training_set_scaled[i*(1+j),0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 5, batch_size = 64)

model.save("./Amodel")

