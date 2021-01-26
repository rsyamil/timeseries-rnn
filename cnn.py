import numpy as np
import util
import keras
from keras.models import Model
from keras.layers import Layer, Flatten, LeakyReLU
from keras.layers import Input, Reshape, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras.layers import Conv1D, UpSampling1D
from keras.layers import AveragePooling1D, MaxPooling1D

from keras.layers import LSTM, Dropout

from keras import backend as K
from keras.engine.base_layer import InputSpec

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy
from keras import regularizers, activations, initializers, constraints
from keras.constraints import Constraint
from keras.callbacks import History, EarlyStopping

from keras.utils import plot_model
from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects

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
    
class CNN:

    def __init__(self, data, n_lag, n_seq, name=[]):
    
        self.name = name
        
        self.data = data
        self.n_lag = n_lag
        self.n_seq = n_seq
        
        self.n_batch = 0
        self.nb_epoch = 0
        
        self.x = []
        self.y = []
        
        self.model = []
        
    def get_cnn_model(self, input, output):
    
        input_data = Input(shape=(input.shape[1], input.shape[2]))
        
        _ = Conv1D(16, 3, padding='same', data_format='channels_last')(input_data)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling1D(2, padding="same")(_)
        
        _ = Conv1D(8, 6, padding='same', data_format='channels_last')(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling1D(2, padding="same")(_)
        
        _ = Flatten()(_)
        _ = Dropout(0.3)(_)
        
        output_data = Dense(output.shape[1])(_)
        
        self.model = Model(input_data, output_data)
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
        self.model.summary()
        plot_model(self.model, to_file='cnn.png')
        
    def train(self, n_batch=1, nb_epoch=100, load=False):
    
        self.n_batch = n_batch
        self.nb_epoch = nb_epoch
        
        x = self.data.iloc[:, 0:self.n_lag].values
        self.x = x.reshape(x.shape[0], x.shape[1], 1)
        self.y = self.data.iloc[:, self.n_lag:].values
        
        self.get_cnn_model(self.x, self.y)
        
        history = History()

        losses = np.zeros([self.nb_epoch, 2])
        
        for i in tqdm(range(self.nb_epoch)):
        
            self.model.fit(self.x, self.y, 
                        epochs=1, batch_size=self.n_batch, verbose=False,
                        shuffle=True, callbacks=[history])

            losses[i, :] = np.squeeze(np.asarray(list(history.history.values())))
            print ("%d [Loss: %f] [Val loss: %f]" % (i, losses[i, 0], losses[i, 1]))
            
            figs = util.plotLosses(losses, name="cnn_losses")

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
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    