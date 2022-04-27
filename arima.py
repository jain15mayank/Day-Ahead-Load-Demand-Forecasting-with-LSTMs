import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot
from copy import copy, deepcopy
import pytz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
from pickle import dump, load

from joblib import Parallel, delayed
import joblib

from matplotlib.dates import num2date
from matplotlib.ticker import Formatter
from datetime import datetime

from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from numpy import log
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import gc
gc.collect()

print('Library Import Complete!')

### IMPORT DATA

scaler = load(open('./data/scaler.pkl', 'rb'))
officeTS = load(open('./data/officeTS.pkl', 'rb'))

trainSplit = load(open('./data/trainSplit.pkl', 'rb'))
testSplit = load(open('./data/testSplit.pkl', 'rb'))
dateTrainSplit = load(open('./data/dateTrainSplit.pkl', 'rb'))
dateTestSplit = load(open('./data/dateTestSplit.pkl', 'rb'))

print('Data Import Complete!')


# Perform Standardization
trainScaled = np.transpose(scaler.transform(np.transpose(trainSplit)))
testScaled = np.transpose(scaler.transform(np.transpose(testSplit)))

print('Shape of train set (scaled)', trainScaled.shape)
print('Shape of test set (scaled)', testScaled.shape)


#********************************************************************************************
# UTILITY FUNCTIONS *************************************************************************
#********************************************************************************************
# Make Windows
def makeWindowsFromContinuousSteps(officesData, dateData, history, leadSteps):
    '''
    officesData     : 2-D Numpy Array (number of offices, total number of readings)
    dateData        : 2-D Numpy Array (number of offices, total number of readings)
    history         : Number of TIMESTEPS to be considered as Input to Forecasting Model
    leadSteps       : Number of TIMESTEPS to be considered as Output to Forecasting Model
    '''
    dataX, dataY, officeNumber = [], [], []
    dateX, dateY = [], []
    idxWindows = []

    # If only 1 office data is present, i.e. shape of officesData is
    # (total number of readings,), then reshape to (1, total number of readings)
    if len(officesData.shape)==1:
        officesData = officesData.reshape(1,-1)
    elif not len(officesData.shape)==2:
        raise('Unacceptable input shape of officesData variable - ' + str(officesData.shape))
    
    # If only 1 date data is present, i.e. shape of dateData is
    # (total number of readings,), then reshape to (1, total number of readings)
    if len(dateData.shape)==1:
        dateData = dateData.reshape(1,-1)
    elif not len(dateData.shape)==2:
        raise('Unacceptable input shape of dateData variable - ' + str(dateData.shape))

    for i in range(officesData.shape[0]):
        continuousSubSequenceStartIndex = 0
        continuousSubSequenceStopIndex  = 0
        searchingForNAN = True
        seqCounter = 0
        while seqCounter<len(officesData[i]):
            if seqCounter==len(officesData[i])-1:
                if np.isnan(officesData[i, seqCounter]):
                    continuousSubSequenceStopIndex = seqCounter - 1
                else:
                    continuousSubSequenceStopIndex = seqCounter
                ### Perform continuousSubSequence Operation - Extract Windows ###
                if continuousSubSequenceStopIndex - continuousSubSequenceStartIndex + 1 >= history + leadSteps:
                    for k in range(continuousSubSequenceStartIndex, continuousSubSequenceStopIndex+1):
                        if k + history + leadSteps <= continuousSubSequenceStopIndex + 1:
                            dataX.append(officesData[i, k:k+history])
                            dataY.append(officesData[i, k+history:k+history+leadSteps])
                            officeNumber.append(i)
                            dateX.append(dateData[i, k:k+history])
                            dateY.append(dateData[i, k+history:k+history+leadSteps])
                            idxWindows.append((i, k))
                break
            if not np.isnan(officesData[i, seqCounter]):
                if not searchingForNAN:
                    searchingForNAN = True
                    continuousSubSequenceStartIndex = seqCounter
            else:
                if searchingForNAN:
                    searchingForNAN = False
                    continuousSubSequenceStopIndex = seqCounter - 1
                    ### Perform continuousSubSequence Operation - Extract Windows ###
                    if continuousSubSequenceStopIndex - continuousSubSequenceStartIndex + 1 >= history + leadSteps:
                        for k in range(continuousSubSequenceStartIndex, continuousSubSequenceStopIndex+1):
                            if k + history + leadSteps <= continuousSubSequenceStopIndex + 1:
                                dataX.append(officesData[i, k:k+history])
                                dataY.append(officesData[i, k+history:k+history+leadSteps])
                                officeNumber.append(i)
                                dateX.append(dateData[i, k:k+history])
                                dateY.append(dateData[i, k+history:k+history+leadSteps])
                                idxWindows.append((i, k))
            seqCounter += 1
    return dataX, dataY, officeNumber, dateX, dateY, idxWindows
    
def calcInvScaledError(scaledPrediction, scaledTruth, errorMetric='RMSE', scaler=None, dataToScalerIndices=None):
    if scaler is not None:
        if dataToScalerIndices is None:
            if len(scaler.mean_) > 1:
                raise("Different scaler used for different datapoints but back reference is not provided - dataToScalerIndices is None!")
            else:
                inv_scaledPrediction = np.transpose(scaler.inverse_transform(np.transpose(scaledPrediction)))
                inv_scaledTruth = np.transpose(scaler.inverse_transform(np.transpose(scaledTruth)))
        else:
            inv_scaledPrediction = deepcopy(scaledPrediction)
            inv_scaledTruth = deepcopy(scaledTruth)
            for i in range(len(scaledPrediction)):
                inv_scaledPrediction[i,:] = (scaledPrediction[i,:]*(scaler.mean_[dataToScalerIndices[i]]))+np.sqrt(scaler.var_[dataToScalerIndices[i]])
                inv_scaledTruth[i,:] = (scaledTruth[i,:]*(scaler.mean_[dataToScalerIndices[i]]))+np.sqrt(scaler.var_[dataToScalerIndices[i]])
        prediction = inv_scaledPrediction
        truth = inv_scaledTruth
    else:
        prediction = scaledPrediction
        truth = scaledTruth
    if errorMetric=='RMSE':
        error = mean_squared_error(truth, prediction, squared=False)
    elif errorMetric=='MSE':
        error = mean_squared_error(truth, prediction, squared=True)
    else:
        raise('Unsupported error metric - ' + errorMetric)
    return error

#********************************************************************************************
#********************************************************************************************

# Selected Working Hours: start_time - end_time (MON-FRI)
start_time = 5
end_time = 19

# Calculate total number of days and readings per day
readingsPerDay = (end_time-start_time)*4
totalDays = int(officeTS.shape[1]/readingsPerDay)
print('Readings Per Day = ', readingsPerDay)
print('Total Number of Days = ', totalDays)

# History and Lead Time
historySteps = 5*readingsPerDay #== 5 Days
leadSteps = 1*readingsPerDay #== 1 Day
print('History Steps = ', historySteps)
print('leadSteps = ', leadSteps)


officeIDX = 2
lastTrainingEntries = 10000

gc.collect()
print('starting to train the model')
model = ARIMA(pd.Series(trainScaled[officeIDX][-lastTrainingEntries:]).dropna(), order=(1,1,2), seasonal_order=(1,0,2,56))#, order=(4,1,8))
model_fit = model.fit()
print(model_fit.summary())

modelForecast = model_fit.forecast(len(testScaled[officeIDX]))
print('Forecast Done')

del model
del model_fit

# save the Forecast
joblib.dump(modelForecast, open('./data/arima_forecast_complete'+str(officeIDX)+'.pkl', 'wb'))
print('Forecast Saved')


# save the Model
#joblib.dump(model_fit, open('./models/arima_model_complete.pkl', 'wb'))
#print('Model Saved')

