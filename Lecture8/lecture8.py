import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

date_range= pd.date_range(start='/1/2020', periods=100, freq='D')
values = np.random.randn(100)

time_series = pd.DataFrame({'date': date_range, 'value': values})
time_series.set_index('date', inplace=True)

#print(time_series.head())s

plt.plot(time_series.index, time_series['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time series data')
plt.show()

time_series_with_nan=time_series.copy()
time_series_with_nan[::10]=np.nan

plt.plot(time_series.index, time_series_with_nan['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time series with NAN data')
plt.show()

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