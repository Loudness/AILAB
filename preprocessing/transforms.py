
import numpy as np


# fills out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for idx in range(len(series)-window_size):
        theEnd = idx+window_size
        newInput = series[idx:theEnd]
        newOutput = series[theEnd]
        y.append(newOutput)
        X.append(newInput)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y
