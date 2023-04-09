import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error

import keras
from IPython.display import clear_output
import matplotlib as mpl

def RMSE(data1, data2):
    return np.sqrt(np.mean((data1.flatten()-data2.flatten())**2))

#scatter plots for training and testing, color by field label
def scatterPlot(data1, data2, xlabel, ylabel, color, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.scatter(data1, data2, c=color, alpha=0.05)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(title)
    ax.annotate(f"RMSE: {RMSE(data1, data2):.3f}", xy=(0.8, 0.8),  xycoords='data',
            xytext=(0.05, 0.9), textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    
#histogram for error for each datapoints
def histPlot(data1, label1, color, title):
    bb = np.linspace(np.min(data1), np.max(data1), 50)
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.hist(data1.flatten(), color=color, alpha=0.4, bins=bb)  
    plt.legend([label1])
    plt.xlim([np.min(data1), np.max(data1)])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.ylabel("Frequency")
    plt.xlabel("RMSE")
    plt.title(title)

def barPlot(data, xlabel, ylabel, color, title):
    timesteps = np.linspace(1, data.shape[0], data.shape[0])
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.bar(timesteps, data, color=color, alpha=0.4)
    plt.ylim([0, np.max(data)*1.2])
    plt.xlim([0, data.shape[0]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def qc_timeseries_stats(y_train, forecasts_train, y_test, forecasts_test, y_noisy, fm_transfer, n_lag, n_seq):
    
    #training and testing data
    scatterPlot(y_train[:, n_lag:], forecasts_train, '$y_{train}$', '$\hat{y}_{train}$', 'r', 'Training data (windows)')
    scatterPlot(y_test[:, n_lag:], forecasts_test, '$y_{test}$', '$\hat{y}_{test}$', 'b', 'Testing data (windows)')
    
    #remove the first n_lag points
    y_noisy = y_noisy[:, n_lag:,:]
    
    #transfer dataset (completely unseen), doesnt include first n_lag points
    n, s, w, c = fm_transfer.shape
    fm_transfer = np.reshape(fm_transfer, [n, s*w, c])[:, 0:y_noisy.shape[1],:]
    scatterPlot(y_noisy, fm_transfer, '$y_{transfer}$', '$\hat{y}_{transfer}$', 'g', 'Transfer data')
    
    #barplot for error by time steps
    errors = evaluate_forecasts(y_noisy, fm_transfer, verbose=False)
    barPlot(errors, 'Timestep', 'RMSE', 'g', 'Transfer data')
    
    #histogram of error for each data point
    n_samples = y_noisy.shape[0]
    errors = np.sqrt(np.mean((np.reshape(y_noisy,[n_samples, -1]) - np.reshape(fm_transfer, [n_samples, -1]))**2, axis=1))
    histPlot(errors, 'Errors', 'g', 'Transfer data - Error')

#quality check timeseries windows
def qc_timeseries_windows(y_train, y_test, n_lag, n_seq):
    train_period = y_train.shape[0]     #time windows
    test_period = y_test.shape[0]
    feat = 0                            #0, 1, 2 for oil, water, gas
    
    fig = plt.figure()
    for i in range(train_period):
        t = np.linspace(0+i, n_lag+n_seq-1+i, n_lag+n_seq)
        plt.scatter(t, y_train[i, :, feat], c='g', alpha=0.1)
	
    for i in range(test_period):
        t = np.linspace(train_period+i, train_period+n_lag+n_seq-1+i, n_lag+n_seq)
        plt.scatter(t, y_test[i, :, feat], c='r', alpha=0.1)
    plt.ylim([0, 1])
    plt.xlabel('Accumulated timeseries windows')
    plt.ylabel('Feature')
    plt.title('QC timeseries windows')
    fig.savefig('readme/qc_timeseries_windows.png')

#calculate error by timestep for all features
def evaluate_forecasts(actual, forecasts, verbose=True):
    errors = np.zeros([forecasts.shape[1],])
    for i in range(forecasts.shape[1]):
        rmse = np.sqrt(mean_squared_error(actual[:, i, :], forecasts[:, i, :]))
        errors[i] = rmse
        if verbose:
            print('t+%d RMSE: %f' % ((i+1), rmse))
    return errors

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

    
