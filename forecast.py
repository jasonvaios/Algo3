import sys
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import *
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
#python3 forecast.py nasd_quer.csv
print("eimai xazh glwssa")
print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")

file=sys.argv[1]
#file = enumerate(sys.argv)#
print(file)
df=pd.read_csv(file)
print('Number of rows and columns:', df.shape)
df.head(5)