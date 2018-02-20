
##################################################
# RNN Lab1: Time series prediction
################################################## 

# The Plan:
# 1. Normalize data 
# 2. Cut the time series into sequences
# 3. Split into training and testing sets
# 4. Buil and run the RNN/LSTM model
# 5. Check model performance, output everything to its own folder for comparison with the other runs. 
# This is my attempt, there are other ways to do it

# Using the spy.csv from the LSTM_stock project as lab file.
# https://www.nyse.com/quote/XXXX:SPY
                                                                  
#/Aries

import os
import time
from dataload import csvloader
from datautils import filehelper
from datautils import graphhelper
from datautils import outputhelper 


from preprocessing import scalers
from preprocessing import transforms
import matplotlib.pyplot as plt
import numpy as np


#Windoze users there is still a bug with seabornin VS own files: If you get NameError: name 'channel' is not defined, see https://github.com/Microsoft/PTVS/pull/3259/files.
# edit C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\Extensions\Microsoft\Python\Core\ptvsd\debugger.py
import seaborn as sns                




def run_lstmLab1():

####################################################################################################
# Part 1. Import data from file
####################################################################################################

                                                   
    #Only get the 'Close' column to the dataframe.
    print("Loading SPY closing prices...")
    inputdatafile = "spy.csv"
    #Get it from data/ folder
    basedata2 = csvloader.csv.csvloader_column(os.path.join("data", inputdatafile),['Close'])
            
    print("Outputting first 5 hits for quick validation of dataFrame")
    print(basedata2.head(5))

####################################################################################################
# Part 2. Normalizing Data
# Clean data is crucial for correct output. 
# If the data contains outliers or missing data, the result could be very misleading.
# First we will pre-process the data by normalizing it with http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# There are different ways to normalize the data. 
# Lets go with the Robustscaler to start with as it is good with large data sets that are not centered around zero.  
####################################################################################################

    #Checkout 
    # http://people.duke.edu/~ccc14/sta-663-2016/05_Machine_Leraning.html

    #Maybe use scaler to something later
    normalizedData, scaler = scalers.getMinMaxScalerData(basedata2)

    #print("Outputting first 5 hits, now normalized, for quick validation")
    #print(normalizedData.head(5))         #only works for pandas dataframe, so convert from numpy.array first.. 

    # lets take a look at our time series, close window to continue
    plt.title('close window to continue')
    plt.plot(normalizedData)
    plt.xlabel('time period')
    plt.ylabel('normalized series value')
    plt.show()

   ####################################################################################################
   # Part 3. Cutting our time series into sequences
   # The data might be rather big and having a loong array is nor optimal or effective for predicting 
   # just a few steps ahead.
   # We should chop it up in "windows" where it learns what the next value(s) is after the window.
   #  
   ####################################################################################################
    #Start taking time
    global_start_time = time.time() 

    WINDOW_SIZE = 10  #Tweak this. Now it is 7 days window.
    X,y = transforms.window_transform_series(series = normalizedData,window_size = WINDOW_SIZE)


    ####################################################################################################
    # Part 4. Splitting into training and testing sets
    # We want to train on one solid chunk of the series (in our case, the first full 2/3 of it), 
    # and validate on a later chunk (the last 1/3) as this simulates how we would predict future values of a time series.
    # NOTE: we are not splitting the dataset randomly as one typically would do when validating a regression model. 
    # This is because our input/output pairs are related temporally. We don't want to validate our model by training on a random subset of the series 
    # and then testing on another random subset, as this simulates the scenario that we receive new points within the timeframe of our training set.
    ####################################################################################################

    #Add below to own module? Yes, at it seems it can be reused. 
    # split our dataset into training / testing sets
    TRAIN_TEST_SPLIT = int(np.ceil(2*len(y)/float(3)))   # set the split point
    print("TRAIN_TEST_SPLIT is " + str(TRAIN_TEST_SPLIT) +  " from a total of " + str(len(y)))

    # partition the training set
    X_train = X[:TRAIN_TEST_SPLIT,:]
    y_train = y[:TRAIN_TEST_SPLIT]

    # keep the last chunk for testing
    X_test = X[TRAIN_TEST_SPLIT:,:]
    y_test = y[TRAIN_TEST_SPLIT:]

    # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
    X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], WINDOW_SIZE, 1)))
    X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], WINDOW_SIZE, 1)))


    ####################################################################################################
    # Part 5. Build and run the RNN/LSTM regression model
    # See the model definition for more details
    ####################################################################################################
        
    #Check why VS cannot resolve models.RNN, even though it is there.
    from models.RNN import sequential

    #Create model to use
    model = sequential.LSTM_seriesMethod1(WINDOW_SIZE)

    # And run it!
    #Play with these number depending on result
    EPOCHS=200   #Up to 1000 is fair to test with.
    BATCH_SIZE=60
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    ####################################################################################################
    # Part 6. Check model performance
    # Results here should be
    # training_error < 0.02
    # testing_error < 0.02
    # if not:  Increasing the number of epochs you take (max 1000 should be enough) 
    # and/or adjusting your batch_size.
    ####################################################################################################

    # generate predictions for training
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # print out training and testing errors
    training_error = model.evaluate(X_train, y_train, verbose=1)
    print('Training error = ' + str(training_error))

    testing_error = model.evaluate(X_test, y_test, verbose=1)
    print('Testing error = ' + str(testing_error))
    

    #Stop taking time
    timeTaken = time.time() - global_start_time

    ####################################################################################################
    # Part 7. Show results!
    # Using helper methods to automagically create a subfolder with an HTML showing results.
    # For each run a new folder is created, so we can compare multiple runs later.
    ####################################################################################################
    
    resultFolder = filehelper.generateNewResultFolder() #creates new foldername based on timestamp and filename
    
    
    #TODO: Add error results (predictions), time taken, computername (os indpendent)
    thoughts = "Yes, I believe that it was a <b>great</b> experiment </BR> "
    thoughts += "Training time in seconds:<B>"  + str(timeTaken)  + "</B></BR>"
    thoughts += "Window size peek:<B>" + str(WINDOW_SIZE) + "</B></BR>"
    thoughts += "Training error:<B>" + str(training_error)  + "</B></BR>"
    thoughts += "Testing error:<B>"  + str(testing_error)  + "</B></BR>"
    thoughts += "Total days input:<B>"  + str(len(normalizedData))  + "</B></BR>"
    thoughts += "Days used for training:<B>"  + str(len(train_predict))  + "</B></BR>"
    thoughts += "Days used for testing:<B>"  + str(len(test_predict))  + "</B></BR>"

  



    
    #Graphs are a bit harder to make generic, so custom for now.
    GRAPHFILENAME = "result.png"
    graphhelper.createGraphLSTM_Lab1(resultFolder + os.sep + GRAPHFILENAME, WINDOW_SIZE, normalizedData, TRAIN_TEST_SPLIT,train_predict, test_predict)
      
    outputhelper.resultToBrowser(resultFolder, GRAPHFILENAME, thoughts)


    
