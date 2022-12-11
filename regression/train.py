import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization, Input, Dense, Dropout

df = pd.read_csv('../train_dataset.csv')
df['price_vnd'] = df['price_vnd'].astype(int)
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

y = df['price_vnd'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)

output_layer = Dense(1)(x)
model = Model(inputs=input_layer, outputs=output_layer)

model.compile( loss='mean_squared_error', optimizer = keras.optimizers.Adam())

def scheduler(epoch, lr):
    if epoch <= 8:
        return 1e-3
    if epoch <= 24:
        return 1e-4
    else:
        return lr * tf.math.exp(-0.1)
schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

BATCH_SIZE = 64
EPOCHS = 200
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode = 'min', verbose = 1)
model_save = tf.keras.callbacks.ModelCheckpoint('model-regression.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[callback, model_save, schedule])