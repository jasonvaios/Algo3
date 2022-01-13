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

#python3 forecast.py nasd_quer.csv

# print(f"Arguments count: {len(sys.argv)}")
# for i, arg in enumerate(sys.argv):
#     print(f"Argument {i:>6}: {arg}")

# file=sys.argv[1]
file = "/content/drive/MyDrive/nasdaq2007_17.csv"
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/nasdaq2007_17.csv", sep="\t")

# file = "/content/nasd_query.csv"

#df=pd.read_csv(file,sep="\t")
n=5
shape = df.shape
print("Number of rows and columns:",shape)
colls=shape[1]
rows=shape[0]
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
# Feature Scaling
#sc = MinMaxScaler(feature_range = (0, lag))
#training_set_scaled = sc.fit_transform(train.reshape(-1,1))

qt = QuantileTransformer()
training_set_scaled=qt.fit_transform(train.reshape(-1,1))
# Compiling the RNN
lag =1
for j in range(1,n+1):
    name=df.iloc[j,0:1].values

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(window, trainsize-1):
        X_train.append(training_set_scaled[j*i-window:j*i,0])
        y_train.append(training_set_scaled[i*j,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 5, batch_size = 32)

    dataset_train = df.iloc[j,1:trainsize]#:
    dataset_test = df.iloc[j,trainsize:]#:
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 1)
    input= df.iloc[j,colls-testsize-window:].values#1
    input=input.reshape(-1,1)
    input=qt.transform(input)
    x_test=[]
    for i in range(window,testsize+window+1):
        x_test.append(input[i-window:i,0])
    x_test=np.array(x_test)
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = qt.inverse_transform(predicted_stock_price)
    dataset_test=dataset_test.to_numpy().reshape(-1,1)

    plt.plot(range(len(dataset_test)),dataset_test[:,0], color = "green", label = "Real "+name+ " Stock Price")
    plt.plot(range(len(predicted_stock_price)),predicted_stock_price[:,0], color = "purple", label = "Predicted "+name+ " Stock Price")
    plt.title(name+" Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(name+" Price")
    plt.legend()
    plt.show()
