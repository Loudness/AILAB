#Some helpers for crating folders (cross-platform) and so on.

import os
import errno
import time
import inspect 

#Create folder if it does not exist.
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# Create a file/foldername based on timestamp
def create_current_filename():
    return time.strftime("%Y%m%d-%H%M%S") 

def generateNewResultFolder():

    #Use timestamp to have separated and unique folder names.
    fldname = create_current_filename()
    
   
    
    #Prepend calling filename, IE decriptive testname for example 'LSTM_Lab1' if called from LSTM_Lab1.py
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__   
    filename = os.path.splitext(filename)[0]
    
    #Better to create all folders in a results folder so we keep it cleaner
    #fldname =   filename  + os.sep + "results" + os.sep +  "result" + fldname 
    fldname =   filename  + os.sep +  "result" +"_" + fldname 

    #Create new folder
    make_sure_path_exists(fldname)
    return fldname

