import pandas as pd
import matplotlib.pyplot as plt  #matplotlib inline
import seaborn as sns
sns.set()
  
df = pd.read_csv('taxi-fares.csv')
df.head()
 
 
import datetime
from math import sqrt 
 
df = df[df['passenger_count'] == 1]
df = df.drop(['key', 'passenger_count'], axis=1)
 
for i, row in df.iterrows():
    dt = datetime.datetime.strptime(row['pickup_datetime'], '%Y-%m-%d %H:%M:%S UTC')
    df.at[i, 'day_of_week'] = dt.weekday()
    df.at[i, 'pickup_time'] = dt.hour
    x = (row['dropoff_longitude'] - row['pickup_longitude']) * 54.6
    y = (row['dropoff_latitude'] - row['pickup_latitude']) * 69.0
    distance = sqrt(x**2 + y**2)
    df.at[i, 'distance'] = distance
 
df.drop(['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)
 
df = df[(df['distance'] > 1.0) & (df['distance'] < 10.0)]
df = df[(df['fare_amount'] > 0.0) & (df['fare_amount'] < 50.0)]
df.head()

#The resulting dataset contains columns for the day of the week (0-6, where 0 corresponds to Monday),
# the hour of day (0-23), and the distance traveled in miles, and from which outliers have been removed: 
 

from keras.models import Sequential
from keras.layers import Dense
 

# create a network with an input layer that accepts three values (day, time, and distance),
# two hidden layers with 512 neurons each, 
# and an output layer with a single neuron (the predicted fare amount).
#Rectified linear units (ReLU) activation function, which, you’ll recall, 
#adds non-linearity by turning negative numbers into 0s
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=3))
model.add(Dense(512, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.summary()

#mean absolute error (MAE) to measure loss


x = df.drop('fare_amount', axis=1)
y = df['fare_amount']
 

#Now separate the feature columns from the label column and use them to train the network.
# Set validation_split to 0.2 to validate the network using 20% of the training data.
# Train for 100 epochs and use a batch size of 100. 
#Given that the dataset contains more than 38,000 samples, 
#this means that about 380 backpropagation passes will be performed in each epoch:
#validation_split=0.2 tells Keras that in each epoch, 
#it should train with 80% of the rows in the dataset and 
#test, or validate, the network’s accuracy with the remaining 20%

hist = model.fit(x, y, validation_split=0.2, epochs=20, batch_size=100)


err = hist.history['mae']  #mean absolute error (MAE) to measure loss
val_err = hist.history['val_mae']
epochs = range(1, len(err) + 1)
 
plt.plot(epochs, err, '-', label='Training MAE')
plt.plot(epochs, val_err, ':', label='Validation MAE')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='upper right')
plt.plot()


from sklearn.metrics import r2_score
print(r2_score(y, model.predict(x)))

#regression score function Best possible score is 1.0


import numpy as np
zz=model.predict(np.array([[4, 17, 2.0]]))
print(zz)
#to hire a taxi for a 3-mile trip at 5:00 p.m. on Friday afternoon


zz=model.predict(np.array([[5, 17, 2]]))
print(zz)
#to hire a taxi for a 2-mile trip at 5:00 p.m. on Saturday afternoon














