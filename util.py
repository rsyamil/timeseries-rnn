import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error

import keras
from IPython.display import clear_output
import matplotlib as mpl

#function to view training and validation losses
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss", c = 'green')
        plt.plot(self.x, self.val_losses, label="val_loss", c = 'red')
        plt.legend()
        plt.show()
        
#function to view multiple losses
def plotAllLosses(loss1, loss2):         
    N, m1f = loss1.shape
    _, m2f = loss2.shape
    
    print(loss1.shape)
    print(loss2.shape)
    
    fig = plt.figure(figsize=(6, 12))
    plt.subplot(2, 1, 1)
    plt.plot(loss1[:, 0], label='loss1_check1')
    plt.plot(loss1[:, 1], label='loss1_check2')
    plt.plot(loss1[:, 2], label='loss1_check3')
    plt.plot(loss1[:, 3], label='loss1_check4')
    plt.plot(loss1[:, 4], label='loss1_check3')
    plt.plot(loss1[:, 5], label='loss1_check4')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(loss2[:, 0], label='loss2_check1')
    plt.plot(loss2[:, 1], label='loss2_check2')
    plt.legend()
    
    return fig

#function to view multiple losses
def plotLosses(loss, name=[]):         
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(loss[:, 0], label='loss', c = 'green')
    plt.legend()
    
    plt.grid(False)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    fig.tight_layout() 
    fig.savefig(name+'.png')
    plt.close(fig)

#calculate error by timestep
def evaluate_forecasts(actual, forecasts):
    for i in range(forecasts.shape[1]):
        rmse = np.sqrt(mean_squared_error(actual[:, i], forecasts[:, i]))
        print('t+%d RMSE: %f' % ((i+1), rmse))

def plot_profiles(ref, actual_train, forecasts_train, actual_test, forecasts_test, n_lag, n_seq, forecasts_test_multi, name):

    timesteps = np.linspace(0, ref.shape[0]-1, ref.shape[0])

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(timesteps, ref, c = 'gray', alpha=0.2, s=10)
    
    for i in range(actual_train.shape[0]):
        t_w = timesteps[i+n_lag:i+n_lag+n_seq]
        plt.scatter(t_w, actual_train[i,:], c = 'red', alpha=0.2, s=10)
    
    for i in range(forecasts_train.shape[0]):
        t_w = timesteps[i+n_lag:i+n_lag+n_seq]
        plt.plot(t_w, forecasts_train[i,:], c = 'red', alpha=0.2)
        
    shift = actual_train.shape[0]
    for i in range(actual_test.shape[0]):
        t_w = timesteps[shift+i+n_lag:shift+i+n_lag+n_seq]
        plt.scatter(t_w, actual_test[i,:], c = 'blue', alpha=0.2, s=10)
        
    for i in range(forecasts_test.shape[0]):
        t_w = timesteps[shift+i+n_lag:shift+i+n_lag+n_seq]
        plt.plot(t_w, forecasts_test[i,:], c = 'blue', alpha=0.2)
        
    #multistep    
    t_w = np.linspace(0, forecasts_test_multi.shape[0]-1, forecasts_test_multi.shape[0]) +shift+0+n_lag
    plt.plot(t_w, forecasts_test_multi, c = 'green', alpha=0.8)
    
    plt.ylim([0, np.max(ref)])
    plt.xlabel("Time")
    plt.ylabel("Oil rate (bpd)")
    
    fig.tight_layout() 
    fig.savefig(name+'.png')
    #plt.close(fig)
    
