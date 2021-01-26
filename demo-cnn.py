import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_style("whitegrid")

import dataloader
import util
import cnn

if __name__ == "__main__":

    n_lag = 6
    n_seq = 6
    ref_case = 3
    
    Bakken = dataloader.DataLoader(verbose=False)
    _, _, _, _, _, _, _, _ = Bakken.load_data()
    y0_train, y0_test = Bakken.get_time_series(ref_case=ref_case, n_lag=n_lag, n_seq=n_seq, split=0.7)

    print(y0_train.shape)
    print(y0_test.shape)
    
    model = cnn.CNN(y0_train, n_lag=n_lag, n_seq=n_seq, name="cnn")
    model.train(n_batch=1, nb_epoch=150, load=False)
    
    forecasts_test = model.forecasts(y0_test)
    forecasts_train = model.forecasts(y0_train)
    
    util.evaluate_forecasts(y0_test.iloc[:, n_lag:].to_numpy(), np.array(forecasts_test))
    
    n_steps = 6
    forecasts_test_multi = model.forecasts_multistep(y0_test, n_steps=n_steps)  
    
    def sc(data):
        min_ = np.min(Bakken.y[ref_case, :])
        max_ = np.max(Bakken.y[ref_case, :])
        return (data * (max_ - min_)) + min_   
        
    util.plot_profiles(sc(Bakken.ref), sc(y0_train.iloc[:, n_lag:].to_numpy()), 
                   sc(np.array(forecasts_train)), sc(y0_test.iloc[:, n_lag:].to_numpy()), 
                   sc(np.array(forecasts_test)), n_lag, n_seq, sc(np.array(forecasts_test_multi).flatten()),
                  "cnn_forecasts")

