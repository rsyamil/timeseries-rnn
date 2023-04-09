import numpy as np
import util
import keras
from keras.models import Model
from keras.layers import Layer, Flatten, LeakyReLU
from keras.layers import Input, Reshape, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras.layers import Conv1D, UpSampling1D
from keras.layers import AveragePooling1D, MaxPooling1D

from keras.layers import LSTM, Dropout, Concatenate, TimeDistributed

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
    
class XRNN:

    def __init__(self, props, data, n_lag, n_seq, name=[]):
    
        self.name = name
        
        self.data = data                #dimension is of (n_lag+n_seq) x n_feature
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_feature = data.shape[-1]
        
        self.n_batch = 0
        self.nb_epoch = 0
        
        self.n_neurons = 50
        
        self.props = props              #dimension is of number of completion/form params
        self.x = []
        self.y = []
        
        self.model = []
        
    def get_rnn_model(self, input, output):
        '''input : n_lag from timeseries
	   output : n_seq from timeseries
	'''
        input_data = Input(batch_shape=(self.n_batch, input.shape[1], input.shape[2]))
        input_props = Input(batch_shape=(self.n_batch, self.props.shape[1]))
        
        dyn_encoding = LSTM(self.n_neurons, stateful=True, return_sequences=False)(input_data)
        
        _ = Dense(5)(input_props)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = Dense(4)(_)
        _ = LeakyReLU(alpha=0.3)(_)
        prop_encoding = Dense(3)(_)
        
        conc = Concatenate(axis=1)([dyn_encoding, prop_encoding])
        _ = Dropout(0.3)(conc)
	
        _ = Dense(self.n_seq*self.n_feature*3)(_)
        _ = Dense(self.n_seq*self.n_feature*2)(_)
        _ = Reshape((self.n_seq, self.n_feature*2))(_)
	
        _ = LSTM(self.n_neurons, stateful=True, return_sequences=True)(_)
        output_data = TimeDistributed(Dense(self.n_feature))(_)
        
        self.model = Model([input_data, input_props], output_data)
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))
        self.model.summary()
        plot_model(self.model, to_file='readme/'+self.name+'_arch.png', show_shapes=True)
        
    def train(self, n_batch=1, nb_epoch=100, load=False):
    
        self.n_batch = n_batch
        self.nb_epoch = nb_epoch
        
        self.x = self.data[:, 0:self.n_lag, :]
        self.y = self.data[:, self.n_lag:, :]
        
        self.get_rnn_model(self.x, self.y)
        
        if not load:
            history = History()
            losses = np.zeros([self.nb_epoch, 2])
        
            for i in tqdm(range(self.nb_epoch)):
                self.model.fit([self.x, self.props], self.y, 
                        epochs=1, batch_size=self.n_batch, verbose=False,
                        shuffle=False, callbacks=[history])
                self.model.reset_states()
            
                losses[i, :] = np.squeeze(np.asarray(list(history.history.values())))
                print ("%d [Loss: %f] [Val loss: %f]" % (i, losses[i, 0], losses[i, 1]))
            
                figs = util.plotLosses(losses, name="readme/"+self.name+"_losses")
            self.model.save('readme/'+self.name+'model.h5')
        else:
            print("Trained model loaded")
            self.model = load_model('readme/'+self.name+'model.h5')
            
    def forecasts(self, props, data):
	#if data is (N, n_lag+n_seq, n_feature), this fn returns (N, n_seq, n_feature)
        forecasts = list()

        for i in range(len(data)):

            x = data[i, 0:self.n_lag]
            x = x.reshape(1, len(x), self.n_feature)
            y = data[i, self.n_lag:, :]
            
            p = props[i:i+1, :]

            y_hat = self.model.predict([x, p], batch_size=self.n_batch)
            
            forecasts.append([x for x in y_hat[0, :, :]])

        return forecasts
        
    def forecast_multistep(self, props0, data0, n_steps=3):
        #multistep forecast for a single well
        forecast_multi = list()
        
        x = data0[0, 0:self.n_lag, :] #use only the first one of the chunk
        x = x.reshape(1, len(x), self.n_feature)
        
        p = props0[0:1, :]

        for i in range(n_steps):
            y_hat = self.model.predict([x, p], batch_size=self.n_batch)
            forecast_multi.append(y_hat[0, :, :]) #output size is (1, n_seq, n_features)
            x = y_hat[:, -self.n_lag:, :] #input size is (1, n_lag, n_features), require that n_seq >= n_lag, and use only the last n_lag from the previous n_seq
        return np.array(forecast_multi) #the output is (n_steps, n_seq, n_feature) where total multistep predicted is n_steps*n_seq (non-overlapping!)
    
    def forecasts_multistep_all(self, props, data, n_samples, n_steps=3):
        #function to generate multistep prediction for all the test input (well by well)
        forecasts_multi = list()
        
        skip = int(props.shape[0]/n_samples)
        
        for i in range(n_samples):
            start = i*skip
            end = start + skip
            f = self.forecast_multistep(props[start:end, :], data[start:end, :, :], n_steps=n_steps)
            forecasts_multi.append(f)
            
        return np.array(forecasts_multi) #output is (n_samples, n_steps, n_seq, n_features)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    