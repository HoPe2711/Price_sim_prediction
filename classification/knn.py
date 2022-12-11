import pandas
import numpy as np
from tensorflow import keras
from keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Model
from sklearn.neighbors import KNeighborsRegressor
import pickle

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
y = np.array(y)

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

model.load_weights('model.h5')
layer_output = model.layers[-3].output
intermediate_model = Model(inputs=model.input,outputs=layer_output)
intermediate_prediction = intermediate_model.predict(X, batch_size = 64)
tmp = np.matrix(intermediate_prediction)
df = pandas.concat([df, pandas.DataFrame(tmp)], axis=1)

def KNN_model(filename, neighbor, label):
    tr = df[df['label'] == label]
    X_train = tr.drop(columns=['label', 'price_vnd', 'sim_number']).values
    y_train = tr['price_vnd'].values
    linear = KNeighborsRegressor(n_neighbors=neighbor)
    linear.fit(X_train, y_train)
    pickle.dump(linear, open(filename, 'wb'))

KNN_model("knn_default_5.sav",5,5)   
KNN_model("knn_default_6.sav",5,6)   
# KNN_model("knn_5.sav",1,5)   
# KNN_model("knn_6.sav",1,6)                             
