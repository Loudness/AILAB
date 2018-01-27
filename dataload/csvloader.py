
#Load data from CSV
import numpy as np


# This file keeps a collection of methods for loading CSV files.
# There are multiple ways of loading CSV in python, so none is right nor wrong.  
# So added a few , so you can pick what you need. Please add your own csv file reader, maybe for a custom format. 
# There are different ways to use these depending if you use dataframes or plain text.
# Sometimes you'll have to read the file in plain CSV the save it the read back in another format. 


#class to keep separate namespaces. 
class csv:

    # Return a plain csv file 
    def csvloader_plain(filename):
        f = open(filename, 'rb').read()
        return f

    # Return file as dataframe 
    # all column values must exist and values must be castable to double
    def csvloader_dataframe(filename):
        import pandas as pd
        df = pd.read_csv(filename, sep=',',header=None)
        return df

    # Return file as dataframe 
    # all column values must exist and values must be castable to double
    # GEt column name from input. This assumes that first line has commaseparated column description.
    #Columns can be expressed as ['col1', 'col3', 'col7'] OR ['col1'] OR [1,3,5] (if you are missing column names)
    def csvloader_column(filename, columns):
        import pandas as pd
        df = pd.read_csv(filename, sep=',', usecols=columns)
        return df

    # Return file as dataframe, does some guessing with empty values and so on.
    # Works withs strings too. 
    def csvloader_rec(filename):
        df =  np.recfromcsv(filename, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
        return df

    def csvloader_datarset(filename):
        dataset = np.loadtxt('datasets/normalized_apple_prices.csv')
        return dataset
