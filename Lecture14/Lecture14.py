#BU KODA ÇOK GÜVENMEYİN, HOCANIN YAZDIĞI KOD DEĞİL      NOT: AMA ÇALIŞIYOR :)))))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("carbon.csv")

X = df[['Volume', 'Weight']]
y = df['CO2']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(2,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))  

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=40,
    validation_data=(X_test, y_test)
)

new_car = np.array([[1370, 1650]])
new_car_scaled = scaler.transform(new_car)

prediction = model.predict(new_car_scaled)

print("Predicted CO2 emission:", prediction[0][0])
