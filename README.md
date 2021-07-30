# Day-Ahead Electricity Load Demand Forecasting

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> Jain, M., AlSkaif, T. and Dev, S.(2021). Day-Ahead Electricity Load Demand Forecasting using Long-Short-Term-Memory. *under review*, 2021

### Executive summary
Consumer level inclusion of smart meters has generated electricity consumption data for individual buildings at high temporal resolution. This has allowed electricity companies to analyze and forecast load demand for individual buildings. In this work, we propose a novel deep learning architecture using bi-directional long-short-term-memory units to perform day-ahead load demand forecasting.

### Code
All codes are written in `python3`.
+ `pre-processing.ipynb`: Performs the pre-processing of the data file. 
+ `LSTM_train.ipynb`: Computes the training of the deep neural net. 
+ `LSTM_test.ipynb`: Computes the final testing of the deep neural net.
