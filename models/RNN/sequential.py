#Different RNN (LSTM) models to use with sequential data aka time series 
# TODO: add a few more methods when ideas arise.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# This is a first implementation of a time series LSTM.
# layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
# layer 2 uses a fully connected module with one unit
# the 'mean_squared_error' is used because we are performing regression here
def LSTM_seriesMethod1(window_size):
    
    INOUT_NEURONS = 5 
    model = Sequential()
    model.add(LSTM(INOUT_NEURONS, input_shape=(window_size,1), return_sequences=False))
    model.add(Dense(1))

    #model.add(Activation('linear') ) # test with this later..  /Aries


    #Compile using Keras default values for RMSProp
    #model.compile(loss="mean_squared_error", optimizer="rmsprop")
    
    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
 
    return model
