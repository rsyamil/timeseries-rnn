import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error

import keras
from IPython.display import clear_output
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12})

def RMSE(data1, data2):
    return np.sqrt(np.mean((data1.flatten()-data2.flatten())**2))

class Plotter:

    def __init__(self, n_samples, n_lag, n_seq, y_train, y_test, y_noisy, f_train, f_test, f_multi):
    
        self.n_samples = n_samples
        self.n_lag = n_lag
        self.n_seq = n_seq
	
        self.y_train = y_train
        self.y_test = y_test
        self.y_noisy = y_noisy

        self.f_train = f_train
        self.f_test = f_test
        self.f_multi = f_multi
	
    def indexer(self, ref_case):
        #there are "n_samples" of wells, so skip for the ones you want
        skip = int(self.y_train.shape[0]/self.n_samples)
        s_train = ref_case*skip
        e_train = s_train + skip

        skip = int(self.y_test.shape[0]/self.n_samples)
        s_test = ref_case*skip
        e_test = s_test + skip
        return s_train, e_train, s_test, e_test
	
    def get_data_to_plot(self, ax, ref_case, feat, name):

        #get start and end index for the ref wells
        s_train, e_train, s_test, e_test = self.indexer(ref_case)
        print(s_train, e_train, s_test, e_test)
	
        #get data to plot for the ref case
        ref = self.y_noisy[ref_case][:, feat]
	
        actual_train = self.y_train[s_train:e_train, self.n_lag:, feat]
        forecasts_train = self.f_train[s_train:e_train, :, feat]
	
        actual_test = self.y_test[s_test:e_test, self.n_lag:, feat]
        forecasts_test = self.f_test[s_test:e_test, :, feat]
	
        forecasts_test_multi = self.f_multi[ref_case, :, :, feat]
	
        print(ref.shape, actual_train.shape, forecasts_train.shape, 
		actual_test.shape, forecasts_test.shape, forecasts_test_multi.shape)
	
        plot_profiles(ax, ref, actual_train, forecasts_train, actual_test, forecasts_test, 
		self.n_lag, self.n_seq, forecasts_test_multi, name)
	
    def plot_profiles_nontransfer(self, ref_case, name=[]):
        features = ["Oil", "Water", "Gas"]

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(features)):
            ax = plt.subplot(3, 1, i+1)
            self.get_data_to_plot(ax, ref_case, feat=i, name=name+features[i])
	    
        fig.tight_layout() 
        fig.savefig('readme/'+name+'.png')

#inline function (for same well, plot train/test timeseries)
def plot_profiles(ax, ref, actual_train, forecasts_train, actual_test, forecasts_test, n_lag, n_seq, forecasts_test_multi, name):

    timesteps = np.linspace(0, ref.shape[0]-1, ref.shape[0])

    ax.scatter(timesteps, ref, c = 'gray', alpha=0.2, s=10)

    for i in range(actual_train.shape[0]):
        t_w = timesteps[i+n_lag:i+n_lag+n_seq]
        ax.scatter(t_w, actual_train[i,:], c = 'red', alpha=0.2, s=10)
    
    for i in range(forecasts_train.shape[0]):
        t_w = timesteps[i+n_lag:i+n_lag+n_seq]
        ax.plot(t_w, forecasts_train[i,:], c = 'red', alpha=0.2)
        
    shift = actual_train.shape[0]
    for i in range(actual_test.shape[0]):
        t_w = timesteps[shift+i+n_lag:shift+i+n_lag+n_seq]
        ax.scatter(t_w, actual_test[i,:], c = 'blue', alpha=0.2, s=10)
        
    for i in range(forecasts_test.shape[0]):
        t_w = timesteps[shift+i+n_lag:shift+i+n_lag+n_seq]
        ax.plot(t_w, forecasts_test[i,:], c = 'blue', alpha=0.2)
        
    #flatten multistep and plot 
    forecasts_test_multi = forecasts_test_multi.flatten()
    t_w = np.linspace(0, forecasts_test_multi.shape[0]-1, forecasts_test_multi.shape[0]) +shift+0+n_lag
    ax.plot(t_w, forecasts_test_multi, c = 'green', alpha=0.8)
    
    ax.set_ylim([0, np.max(ref)+0.05])
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rate (bpd)")
    ax.set_title(name)

#inline function (for transfer wells)
def plot_profiles_transfer(ref, n_lag, n_seq, forecasts_test_multi, name):

    timesteps = np.linspace(0, ref.shape[0]-1, ref.shape[0])
    features = ["Oil", "Water", "Gas"]
    clrs = ["g", "b", "r"]
    
    fig = plt.figure(figsize=(14, 4))
    for i in range(len(features)):
        plt.subplot(1, 3, i+1)
        plt.scatter(timesteps, ref[:, i], c = 'gray', alpha=0.2, s=10, label="Reference")
        plt.scatter(timesteps[:n_lag], ref[:n_lag, i], c = clrs[i], alpha=1.0, s=10, label="Early prod.")
        
        #multistep
        fm = forecasts_test_multi[:, :, i].flatten()
        t_w = np.linspace(0, fm.shape[0]-1, fm.shape[0]) +0+n_lag
        plt.plot(t_w, fm, c = clrs[i], alpha=0.8, label=features[i]+" forecast")
	
        #calculate RMSE between ref and forecast (by feature)
        ref_ = ref[n_lag:,i]                      #remove the first n_lag points
        s, w, c = forecasts_test_multi.shape      #reshape, this doesnt include first n_lag points!
        fm_ = np.reshape(forecasts_test_multi, [s*w, c])[0:ref_.shape[0], i]

        plt.ylim([0, 1])
        plt.xlabel("Timesteps")
        plt.ylabel("Rate (bpd)")
        plt.title(features[i]+" forecast"+f" RMSE: {RMSE(ref_, fm_):.3f}")
        plt.legend()
    fig.tight_layout() 
    fig.savefig('readme/'+name+'.png')
