# Day-Ahead Electricity Load Demand Forecasting

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> Jain, M., AlSkaif, T. and Dev, S.(2022). Analyzing Deep Learning Models for Day-Ahead Electricity Load Demand Forecasting. *under review*, 2022

### Executive summary
Consumer level inclusion of smart meters has generated electricity consumption data for individual buildings at high temporal resolution. This has allowed electricity companies to analyze and forecast load demand for individual buildings. In this work, we compare the performance of LSTM based architectures with traditional machine learning and statistical techniques like Random Forests, and SARIMA.

### Code
All codes are written in `python3`.
+ `pre-processing.ipynb`: Performs the pre-processing of the data file. 
+ `arima.py`: Trains the SARIMA method over the training set for a given building and then stores the forecast for the entire test phase. 
+ `randomForest.ipynb`: Trains and stores the random forest model for day-ahead forecasting.
+ `LSTM_train.ipynb`: Trains the LSTM network and stores its weights.
+ `BiLSTM_train.ipynb`: Trains the Bi-LSTM network and stores its weights.
+ `combinedTest.ipynb`: Loads all the models and/or forecasts from before and compute RMSEs or create visualizations for subjective evaluation.

Note: Random Forest Models are not uploaded due to high file size
