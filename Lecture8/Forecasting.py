

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

quotes=pd.read_csv("IBM.csv")

data=quotes[['Date','Close']]

data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')
del data['Date']

all_s=data.describe()
print(all_s)

import seaborn as sns
sns.set()
plt.ylabel('IBM Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(data.index, data['Close'])


train = data[data.index < pd.to_datetime("2023-04-10", format='%Y-%m-%d')]
test = data[data.index > pd.to_datetime("2023-04-10", format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('IBM Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for IBM Data")




from statsmodels.tsa.statespace.sarimax import SARIMAX

y = train['Close']
ARMAmodel = SARIMAX(y, order = (1, 0, 1))
ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 

plt.plot(y_pred_out, color='green', label = 'ARMA Predictions')



from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["Close"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)


from statsmodels.tsa.arima.model import ARIMA


ARIMAmodel = ARIMA(y, order = (1, 0, 1))  #replace 2,3,2
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Blue', label = 'ARIMA Predictions')




arma_rmse = np.sqrt(mean_squared_error(test["Close"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)


SARIMAXmodel = SARIMAX(y, order = (4, 1, 1), seasonal_order=(1,2,3,6))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Pink', label = 'SARIMA Predictions')
plt.legend()

plt.show()
