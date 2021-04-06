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
        
        self.y = []
 
        self.n_lag = 3
        self.n_seq = 3
        
        self.ref = []
    
    def load_data(self, load = "linear"):
        if load == "linear":
            self.y = np.load("Data_Simulated_Bakken/y1.npy").T
            #self.y = self.y[:, 0:100]
        elif load == "hyperbolic":
            self.y = np.load("Data_Simulated_Bakken/y2.npy").T
            #self.y = self.y[:, 0:100]
        elif load == "both":
            y1 = np.load("Data_Simulated_Bakken/y1.npy").T
            y2 = np.load("Data_Simulated_Bakken/y2.npy").T
            self.y = np.concatenate((y1, y2))
            #self.y = self.y[:, 0:100]
    
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
        
        #self.ref = ref + np.random.normal(loc=0.0, scale=0.03, size=ref.shape)
        self.ref = ref
        
        data = self.series_to_supervised(self.ref.tolist(), self.n_lag, self.n_seq)
        
        split_idx = int(data.shape[0]*split)
        y_train_sim, y_test_sim = data[0:split_idx], data[split_idx:]
        
        return y_train_sim, y_test_sim

    

    
    
    