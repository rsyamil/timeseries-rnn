import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import scipy.io as sio
import math
import cv2
import pandas as pd
import subprocess
import matplotlib as m

class DataLoader:

    def __init__(self, verbose=False):

        self.verbose = verbose
        
        self.x = []
        self.y = []
        
        self.cumm = []
        self.label = []
        
        self.n_lag = 3
        self.n_seq = 3
        
        self.ref = []
    
    def load_data(self):
    
        df = pd.read_csv("Data_Simulated_Bakken/DATA_BAKKEN.csv")
        df = df.to_numpy()

        x = df[:, 0:6]
        self.y = df[:, 6:6+60]

        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        self.x = (x - x_min)/(x_max - x_min)
    
        self.cumm = (np.sum(self.y, axis=-1)).flatten()
        
        #derive label for performance
        partitions = [2100, 3700]
        self.label = np.zeros(self.cumm.shape, dtype=np.int16)
        self.label = np.where(self.cumm > partitions[0], 1, self.label)
        self.label = np.where(self.cumm > partitions[1], 2, self.label)
        
        p = 400
        x_train_sim = self.x[0:p, :]
        x_test_sim = self.x[p:, :] 

        train1 = self.cumm[0:p,]
        test1 = self.cumm[p:,] 

        train2 = self.label[0:p,]
        test2 = self.label[p:,] 

        from keras.utils import to_categorical
        train2_one_hot = to_categorical(train2)
        test2_one_hot = to_categorical(test2)
        label_one_hot = to_categorical(self.label)
        
        if self.verbose:
            #visualize the histograms
            plt.figure(figsize=[12, 3.5])
            plt.subplot(1, 3, 1)
            plt.hist(self.cumm, bins=50)
            for p in partitions:
                plt.axvline(x=p, c='r', lw=3)
            plt.xlim([0, 11000])
            plt.title("Data")

            plt.subplot(1, 3, 2)
            #visualize the histograms (CDF)
            plt.hist(self.cumm, bins=50, cumulative=True, density=True)
            for p in partitions:
                plt.axvline(x=p, c='r', lw=3)
            plt.xlim([0, 11000])
            plt.title("Data")

            plt.subplot(1, 3, 3)
            plt.hist(self.label)
            plt.title("Data")
            
        return x_train_sim, x_test_sim, train1, test1, train2, test2, train2_one_hot, test2_one_hot
    
    def series_to_supervised(self, data, n_lag=1, n_seq=1, dropnan=True):
      
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        
        for i in range(n_lag, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
        for i in range(0, n_seq):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def get_time_series(self, ref_case=0, n_lag=1, n_seq=1, split=0.8):
    
        self.n_lag, self.n_seq = n_lag, n_seq
        
        np.random.seed(99)
        
        ref = (self.y[ref_case, :] - np.min(self.y[ref_case, :])) / (np.max(self.y[ref_case, :]) - np.min(self.y[ref_case, :]))
        
        self.ref = ref + np.random.normal(loc=0.0, scale=0.03, size=ref.shape)
        
        data = self.series_to_supervised(self.ref.tolist(), self.n_lag, self.n_seq)
        
        split_idx = int(data.shape[0]*split)
        y_train_sim, y_test_sim = data[0:split_idx], data[split_idx:]
        
        return y_train_sim, y_test_sim

    

    
    
    