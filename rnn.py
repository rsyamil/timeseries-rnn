import numpy as np
import util

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Flatten, LeakyReLU
from tensorflow.keras.layers import Input, Reshape, Dense, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.keras.layers import Conv1D, UpSampling1D
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D

from tensorflow.keras.layers import LSTM, Dropout

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import regularizers, activations, initializers, constraints
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.callbacks import History, EarlyStopping

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from tensorflow.python.keras.utils.generic_utils import get_custom_objects

import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import matplotlib.cm as cm
from matplotlib.colors import Normalize

def RMSE(x, y):
    return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))
    
class RNN:

    def __init__(self, data, n_lag, n_seq, name=[]):
    
        self.name = name
        
        self.data = data
        self.n_lag = n_lag
        self.n_seq = n_seq
        
        self.n_batch = 0
        self.nb_epoch = 0
        
        self.n_neurons = 10
        
        self.x = []
        self.y = []
        
        self.model = []
        
    def get_rnn_model(self, input, output):
    
        input_data = Input(batch_shape=(self.n_batch, input.shape[1], input.shape[2]))
        
        _ = LSTM(self.n_neurons, stateful=True, return_sequences=False)(input_data)
        _ = Dropout(0.3)(_)
        output_data = Dense(output.shape[1])(_)
        
        self.model = Model(input_data, output_data)
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-3))
        self.model.summary()
        plot_model(self.model, to_file='rnn.png')
        
    def train(self, n_batch=1, nb_epoch=100, load=False):
    
        self.n_batch = n_batch
        self.nb_epoch = nb_epoch
        
        x = self.data.iloc[:, 0:self.n_lag].values
        self.x = x.reshape(x.shape[0], x.shape[1], 1)
        self.y = self.data.iloc[:, self.n_lag:].values
        
        self.get_rnn_model(self.x, self.y)
        
        history = History()
        
        losses = np.zeros([self.nb_epoch, 2])
        
        for i in tqdm(range(self.nb_epoch)):
            self.model.fit(self.x, self.y, 
                        epochs=1, batch_size=self.n_batch, verbose=False,
                        shuffle=False, callbacks=[history])
            self.model.reset_states()
            
            losses[i, :]  = np.asarray(list(history.history.values()))[:, i]
            print ("%d [Loss: %f] [Val loss: %f]" % (i, losses[i, 0], losses[i, 1]))
            
            figs = util.plotLosses(losses, name="rnn_losses")
        
    def forecasts(self, test_data):
    
        forecasts = list()
        
        for i in range(len(test_data)):
        
            x = test_data.iloc[i, 0:self.n_lag].values
            x = x.reshape(1, len(x), 1)
            y = test_data.iloc[i, self.n_lag:].values
            
            y_hat = self.model.predict(x, batch_size=self.n_batch)
            forecasts.append([x for x in y_hat[0, :]])
            
        return forecasts
        
    def forecasts_multistep(self, test_data, n_steps=3):
    
        n_steps = n_steps
        forecasts_multi = list()

        x = test_data.iloc[0, 0:self.n_lag].values
        x = x.reshape(1, len(x), 1)

        for i in range(n_steps):

            y_hat = self.model.predict(x, batch_size=self.n_batch)
            forecasts_multi.append(y_hat[0, :])
            x = y_hat.reshape(1, y_hat.shape[1], 1)
            
        return forecasts_multi
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    