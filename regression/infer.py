import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization, Input, Dense, Dropout

df = pd.read_csv('../test.csv')
df['sim_number'] = df['sim_number'].astype(str)

X = []
for i in df['sim_number']:
    b = np.zeros(shape=(9, 10))
    de = 0
    for p in i: 
        a = np.zeros(shape=(10))
        a[int(p)] = 1
        b[de] = a 
        de += 1
    X.append(b)
X = np.array(X)

input_layer = Input(shape=(9, 10))
conv1 = Conv1D(64, 2, activation='relu', padding='same')(input_layer)
conv2 = Conv1D(128, 2, activation='relu', padding='same')(conv1)
x = BatchNormalization()(conv2)
x = MaxPooling1D(2)(x)

lstm1 = Bidirectional(LSTM(200, return_sequences=True), 
                             input_shape=(x.shape[1:]))(x)
x = Dropout(0.2)(lstm1)

lstm2 = Bidirectional(LSTM(200))(x)
x = Dropout(0.2)(lstm2)
x = Dense(256, activation='relu')(lstm2)
x = Dense(64, activation='relu')(x)

output_layer = Dense(1)(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile( loss='mean_squared_error', optimizer = keras.optimizers.Adam())

model.load_weights('model-regression.h5')
pred = model.predict(X, batch_size=64)

pred = [round(item) for item in pred.reshape(pred.shape[0]).tolist()]
df['price_vnd'] = pred
first_column = df.pop('price_vnd')
df.insert(0, 'price_vnd', first_column)

df.to_csv('CNN+LSTM_regression.csv', index = False)