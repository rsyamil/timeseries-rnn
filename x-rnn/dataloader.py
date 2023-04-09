import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as m

class DataLoader:

    def __init__(self, n_samples=10, n_lag=3, n_seq=6, verbose=False):

        self.verbose = verbose
        
        self.x = []         #(600, 6)
        self.y = []         #(600, 60, 3)

        self.x_raw = []         #(600, 6)
        self.y_raw = []         #(600, 60, 3)
        
        self.n_lag = n_lag
        self.n_seq = n_seq
        
        self.y_noisy = []   #length of list depends on "n_samples"
        
        self.n_samples = n_samples
        
        self.x_min = 0
        self.x_max = 0
        self.y_min = np.array ([])   #(3,) for each channel
        self.y_max = np.array ([])   #(3,)
        
    def normalize_x(self):
        self.x_min = np.min(self.x, axis=0)
        self.x_max = np.max(self.x, axis=0)
        self.x = (self.x - self.x_min)/(self.x_max - self.x_min)
    
    def normalize_y(self):
        '''normalize by channel'''
        n_features = self.y.shape[-1]
        for f in range(n_features):
            self.y_min = np.append(self.y_min, np.min(self.y[:,:,f]))
            self.y_max = np.append(self.y_max, np.max(self.y[:,:,f]))
            self.y[:,:,f] = (self.y[:,:,f] - self.y_min[f])/(self.y_max[f] - self.y_min[f])

    def create_shutins(self):
        '''create control vector u = [N, timesteps]
           sample start_shutins = [timesteps+10, timesteps-10]
           sample len_shutins = [3, 6]
           shift production data by len_shutins at start_shutins
           truncate anything more than 60 months
        '''
        pass

    def load_data(self, flip=False):
    
        df = pd.read_csv("Data_Simulated_Bakken/DATA_BAKKEN.csv")
        df = df.to_numpy()

        #original data
        self.x = df[:, 0:6]
        oil = df[:, 6:66]
        water = df[:, 68:128]
        gas = df[:, 128:188]
        self.y = np.stack((oil, water, gas), axis=2)
	
        #create synthetic shut ins
	
        #shuffle x and y together, since theyre from the same provenance! 
        np.random.seed(77)
        shuffle_idx = np.random.permutation(self.x.shape[0])
	
        #shuffle data
        self.x = self.x[shuffle_idx]
        self.y = self.y[shuffle_idx]

	#make copies (for spatial plotting) and normalize
        self.x_raw = np.copy(self.x)
        self.normalize_x()
        
        self.y_raw = np.copy(self.y)
        self.normalize_y()
	
        if flip:
            self.x = np.flip(self.x, axis=0)
            self.y = np.flip(self.y, axis=0)
            print("x and y flipped")
        
    def series_to_supervised(self, data, dropnan=True):
      
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        
        for i in range(self.n_lag, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
        for i in range(0, self.n_seq):
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

    def get_time_series(self, ref_case=0, split=0.8):
        #add noise
        np.random.seed(ref_case)
        ref = self.y[ref_case, :, :]
        ref = ref + np.random.normal(loc=0.0, scale=0.02, size=ref.shape)
        self.y_noisy.append(ref)
        #reshape so 3 features (oil, gas, water) are in the last channel 
        data = self.series_to_supervised(ref).to_numpy()
        data = np.reshape(data, (data.shape[0], 3, self.n_lag+self.n_seq), order='F')
        data = np.swapaxes(data, 1, 2)
        #split data into training and testing
        split_idx = int(data.shape[0]*split)
        y_train_sim, y_test_sim = data[0:split_idx], data[split_idx:]
        #duplicate properties for each time window, and split properties 
        x_train_sim = np.zeros((y_train_sim.shape[0], self.x.shape[1])) + self.x[ref_case, :]
        x_test_sim = np.zeros((y_test_sim.shape[0], self.x.shape[1])) + self.x[ref_case, :]
        return x_train_sim, x_test_sim, y_train_sim, y_test_sim
        
    def get_time_series_xrnn(self, split=0.8):
    
        x_train, x_test, y_train, y_test = self.get_time_series(0, split=split)
      
        for iref in range(1, self.n_samples):
            x0_train, x0_test, y0_train, y0_test = self.get_time_series(iref, split=split)
            x_train = np.append(x_train, x0_train, axis=0)
            x_test = np.append(x_test, x0_test, axis=0)
            y_train = np.append(y_train, y0_train, axis=0)
            y_test = np.append(y_test, y0_test, axis=0)
            
        return x_train, x_test, y_train, y_test
    
        

    
    
    