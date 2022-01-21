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
window=60
# file = "/content/drive/MyDrive/nasdaq2007_17.csv"
# df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/nasdaq2007_17.csv", sep="\t")
file=sys.argv[2]
df=pd.read_csv(file,sep="\t")
shape = df.shape
rows=shape[0]
colls=shape[1]
trainsize=((80*colls)//100)
testsize=((20*colls)//100)
n=int(sys.argv[4])

mae=float(sys.argv[6])
train = df.iloc[:,1:trainsize].values
qt = QuantileTransformer()

training_set_scaled=qt.fit_transform(train.reshape(-1,1))

if(n>rows):
    n=rows

# n=1
model = keras.models.load_model("./Bmodel")
for j in range(n):

    input= df.iloc[j,colls-testsize-window:].values#1
    input=input.reshape(-1,1)
    input=qt.transform(input)
    x_test=[]
    y_test=[]
    for i in range(window,len(input)):
        x_test.append(input[i-window:i,0])
        y_test.append(input[i,0])
        
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    y_test=y_test.reshape(-1,1)
    #x_test=x_test.reshape(-1,1)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    X_test_pred = model.predict(x_test)

    test_mae_loss = np.mean(np.abs(X_test_pred - x_test), axis=1)
    # test_mae_loss = qt.inverse_transform(test_mae_loss)
    # print(test_mae_loss)
    #test_score_df = pd.DataFrame(index=)
    #test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
    test_score_df = pd.DataFrame(index=range(testsize))
    test_score_df['loss'] = test_mae_loss

    test_score_df['threshold'] = mae

    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['close'] = qt.inverse_transform(input[window:])
    anomalies = test_score_df[test_score_df.anomaly == True]
    anomalies.head()
    
    print("ep",anomalies.shape)
   
    plt.plot(test_score_df.index, test_score_df.loss, label='loss')
    plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    plt.xticks(rotation=25)
    plt.legend();

    plt.plot(
        range(len(input)-window), 
        qt.inverse_transform(input[window:]), 
        label='close price'
    );
    # anomalies = np.reshape(anomalies, (x_test.shape[0], x_test.shape[1], 1))
    # print("ADASDASD", kwstas = anomalies.close.to_numpy().reshape(-1,1).shape)
    # np.squeeze(anomalies)
    sns.scatterplot(
        anomalies.index,
        anomalies.close,
        color=sns.color_palette()[3],
        s=52,
        label='anomaly'
    )
    plt.xticks(rotation=25)
    plt.legend();
