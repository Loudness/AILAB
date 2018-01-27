
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

#example comparison
#http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

# Why use a scaler?
# It does effectively narrow the range of values,  some would call it normalizing. 
# for example an array with values from -234 to +21 
# can be narrowed down to range between -1 and +1 (or 0 to 1) and so on.
# Also ouliers (spikes) in the data are also removed, smoothing the dataset for us. 


#Normalizes data to range between 0 and 1
def getMinMaxScalerData(dataset):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        return dataset, scaler

#TODO: Add Robustscaler and standardscaler
