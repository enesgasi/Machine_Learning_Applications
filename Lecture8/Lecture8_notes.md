# Lecture 8

## TIME SERIES
Time series is a sequence of data points collected or recorded at specific time intervals.
The primary characteristic of time series data is its temporal order. This means that the sequence of the recorded data is important.

## Characteristics of Time Series Data

### Trend
this is the long term movement or direction.

### Seasonality
These are patterns that repeat at regular intervals. Such as higher ice cream sales during summer.

### Cyclic Patterns
These do not have a fixed period. These patterns can be affected from economic cycles or other factors, such as outbreaks

### Irregular Components
These are random or unpredictable variations in the data

## Types of Time Series

### Univariate vs Multivariate
Uni: Single variable recorded over time
Multi: Multiple variables recorder over time. 

### Regular vs Irregular
Regular: Data points are recorded at consistent time intervals

Irregular:
Data points are recorded at inconsistent time intervals

## Preprocessing Time Series Data
Before diving into analysis and forecasting, you should preprocess your time series data to ensure acuracy and reliabilty. With ways such as:

1) Data collection and cleaning
2) Handling missing values and outliers
3) Data Transformation by smoothing, differencing and scaling/normalization

## Sequence Prediction Models

### ARMA (Autoregressive Moving Average)
Its a statistical model that predicts future values using past values. It doesnt capture seasonal trends and it assumes data is stationary.

### ARIMA (Autoregressive Integrated Moving Average)
Its an extension to ARMA. It does not assume stationarity but still doesnt capture seasonality.

### SARIMA (Seasonal Autoregressive Integrated Moving Average)
This is the model that can work with non-stationary data and capture some seasonality.


