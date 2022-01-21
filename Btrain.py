import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
import seaborn as sns
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

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#python3 forecast.py nasd_quer.csv

# print(f"Arguments count: {len(sys.argv)}")
# for i, arg in enumerate(sys.argv):
#     print(f"Argument {i:>6}: {arg}")

file=sys.argv[1]
df=pd.read_csv(file,sep="\t")


# file = "/content/drive/MyDrive/nasdaq2007_17.csv"
# df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/nasdaq2007_17.csv", sep="\t")
shape=df.shape
print("Number of rows and columns:",shape)
colls=shape[1]
rows=shape[0]
window=60
n=rows
print("Number of rows:",rows)
print("Number of colls:",colls)

trainsize=((80*colls)//100)
testsize=((20*colls)//100)

print("trainsize:",trainsize)
print("testsize:",testsize)
model = keras.Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 64,  input_shape = (window, 1)))
model.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation
model.add(keras.layers.RepeatVector(n=window))
model.add(LSTM(units = 64, return_sequences = True))
model.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1)))
model.compile(optimizer = 'adam', loss = 'mae')

train = df.iloc[:,1:trainsize].values
test = df.iloc[:,trainsize:].values

qt = QuantileTransformer()
training_set_scaled=qt.fit_transform(train.reshape(-1,1))
# Compiling the RNN
lag =1
TIME_STEPS = window

for j in range(0,n):
    name=df.iloc[j,0:1].values

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(window, trainsize-1):
        X_train.append(training_set_scaled[(j+1)*i-window:(j+1)*i,0])
        y_train.append(training_set_scaled[i*(j+1),0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Fitting the RNN to the Training set
    history = model.fit(X_train, y_train,epochs=5,batch_size=32,validation_split=0.1,shuffle=False)
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend();

    X_train_pred = model.predict(X_train)
    print(X_train_pred.shape)
    print(y_train.shape)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
    print(train_mae_loss.shape)



model.save("./Bmodel")


  #  predicted_stock_price = qt.inverse_transform(predicted_stock_price)
   # dataset_test=dataset_test.to_numpy().reshape(-1,1)

    # plt.plot(range(len(dataset_test)),dataset_test[:,0], color = "green", label = "Real "+name+ " Stock Price")
    # plt.plot(range(len(predicted_stock_price)),predicted_stock_price[:,0], color = "purple", label = "Predicted "+name+ " Stock Price")
    # plt.title(name+" Price Prediction")
    # plt.xlabel("Time")
    # plt.ylabel(name+" Price")
    # plt.legend()
    # plt.show()
