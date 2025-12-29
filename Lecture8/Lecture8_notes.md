# Lecture 8

## Time Series
Time series is a sequence of data points collected or recorded at specific time intervals.
The primary characteristic of time series data is its temporal order. This means that the sequence of the recorded data is important.

## Characteristics of Time Series Data
### Trend
this is the long term movement or direction. For example, general increase in a company's sales over several years

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

Preprocessing Code Example:

```
#FILL NAN POINTS
time_series_filled = time_series_with_nan.fillna(method='ffill')
time_series_filled[:1]=1
plt.plot(time_series_filled.index, time_series_filled['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time series with filled data')
plt.show()

#REMOVE OUTLIERS
from scipy.stats import zscore
z_scores = zscore(time_series_filled)
abs_z_scores = np.abs(z_scores)
filtered_entries=(abs_z_scores<2)
time_series_no_outliers = time_series_filled[filtered_entries]
plt.plot(time_series_no_outliers.index, time_series_no_outliers['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time series with no outlier data')
plt.show()

#MOVING AVERAGE FOR SMOOTHING

moving_avg = time_series_no_outliers.rolling(window=5).mean()
plt.plot(time_series_no_outliers.index, moving_avg['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time series - Moving Average')
plt.show()

#DIFFERENCING
differenced_series=time_series_no_outliers.diff().dropna()
plt.plot(time_series_no_outliers.index[:len(differenced_series)], differenced_series['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time series - Differencing')
plt.show()

#SCALING
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_series= scaler.fit_transform(time_series_no_outliers.values)
plt.plot(time_series_no_outliers.index, scaled_series['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time series - Scaling')
plt.show()
```

## Sequence Prediction Models
### ARMA (Autoregressive Moving Average)
Its a statistical model that predicts future values using past values. But it is kinda flawed, because it doesnt capture seasonal trends and it assumes data is stationary. It assumes that statistical properties wouldnt change over time. These kinds of assumptions does not hold in practice.

### ARIMA (Autoregressive Integrated Moving Average)
Its an extension to ARMA. It does not assume stationarity but still doesnt capture seasonality.

### SARIMA (Seasonal Autoregressive Integrated Moving Average)
This is the model that can work with non-stationary data and capture some seasonality.


