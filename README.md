# timeseries forecasting

This is an example of how to use a 1D convolutional neural network (1D-CNN) and a recurrent neural network (RNN) with long-short-term memory (LSTM) cell for one-step and multi-step timeseries prediction/forecasting. To run:

`python3 <demo-cnn.py|demo-rnn.py>`

The dataset we will use is a simple hyperbolic curve (timeseries) with added Gaussian noise. Technically, the timeseries can be modeled with just three parameters in a hyperbolic function but we use a simple curve just as a demonstration. We split the timeseries into chunks of input-output window to frame the problem as a supervised machine learning problem. We want to predict a time window **y_w+1** of length `n_seq` using the past information window **y_w** of length `n_lag`. 

![dataset](/readme/dataset.jpg)

The windows (i.e. the rows in the table above) now represent our dataset and we split the dataset into a training set and a testing set. We want to learn **f**, which is a time-invariant predictive model that relates **y_w** to **y_w+1**. In this example, we compare a 1D-CNN and an RNN as **f**. 

## 1D-CNN forecast model

The 1D-CNN model has one-dimensional convolution filters that stride the timeseries to extract temporal features. A couple of layers is used to handle some nonlinearities in the data and the simple 1D-CNN model only has 942 parameters. 

![cnn1d_arch](/readme/cnn1d_arch.jpg)

The figure below shows the original timeseries in light-gray scatter points. The training and testing data points (i.e. **y_w+1** only) are shown as red and blue scatter points respectively. The red and blue lines are the forecasts from the 1D-CNN model. The green line represents the multi-step prediction, where previous forecast are fed into the 1D-CNN model in a recursive way. 

![cnn1d_forecasts](/readme/cnn1d_forecasts.jpg)

## RNN (LSTM) forecast model

For the RNN model, we will use an LSTM cell to extract the temporal features, followed by a Dense layer to reshape the LSTM output tensor into the appropriate output size, of length `n_seq`. 

![rnn_arch](/readme/rnn_arch.jpg)

The RNN predictive model has only 546 parameters where 480 parameters belong to the single LSTM cell as shown below. 

![params_nodim](/readme/params_nodim.jpg)

Note that the single LSTM cell stride the input **y_w** of length `n_lag` one point at a time to produce an output of length 10 (in this example). If `return_sequences=True` then the output of each stride will be returned, i.e. instead of output of length 10, the output will take a shape of (6, 10) - one output of length 10 for 6 strides across the entire length of `n_lag`. This will be important later for other applications. 

![rnn_forecasts](/readme/rnn_forecasts.jpg)

The forecasts are shown above and the legends are the same as the 1D-CNN plot in the previous section. 

![comp_cnn1d_rnn](/readme/comp_cnn1d_rnn.jpg)

In the plots above, we compare the multi-step prediction from the 1D-CNN and RNN models. The single-window forecasts (i.e. use observed **y_w** to predict **y_w+1**) for the training and testing sets are similar for the two models. The RNN model however outperforms the 1D-CNN model for multi-step recursive forecasts.  
