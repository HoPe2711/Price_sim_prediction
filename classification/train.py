import pandas
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix

df = pandas.read_csv('../train_dataset.csv')
df['price_vnd'] = df['price_vnd'].astype(int)
df['sim_number'] = df['sim_number'].astype(str)

label = []
for i in df['price_vnd']:
    if i <= 450000:
        label.append(0)
    elif i <= 750000:
        label.append(1)
    elif i <= 1500000:
        label.append(2)
    elif i <= 4000000:
        label.append(3)
    elif i <= 7500000:
        label.append(4)
    elif i <= 30000000:
        label.append(5)
    else:
        label.append(6)
df['label'] = label

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

y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.models.Sequential()

model.add(Conv1D(64, 2, activation='relu', padding='same', input_shape=(9,10)))
model.add(Conv1D(128, 2, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(Bidirectional(LSTM(200, return_sequences=True),
                             input_shape=(9,10)))

model.add(keras.layers.Dropout(0.2))

model.add(Bidirectional(LSTM(200)))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(7, activation='softmax'))

model.compile( loss='sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics=['accuracy'] )

def scheduler(epoch, lr):
    if epoch <= 8:
        return 1e-3
    if epoch <= 24:
        return 1e-4
    else:
        return lr * tf.math.exp(-0.1)
schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, mode = 'max', verbose = 1)
model_save = tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
history = model.fit(X_train, y_train, epochs=200, batch_size = 64, validation_data=(X_test, y_test), callbacks=[callback, model_save, schedule])

model.load_weights('model.h5')
aa = model.predict(X_test)
y_pred = np.argmax(aa, axis=1)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))