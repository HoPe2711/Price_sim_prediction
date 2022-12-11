import pandas
import numpy as np
from tensorflow import keras
from keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Model
import pickle

df = pandas.read_csv('../test.csv')
df['sim_number'] = df['sim_number'].astype(str)
df1 = df

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
aa = model.predict(X, batch_size = 64)
y_pred = np.argmax(aa, axis=1)
df['pred'] = y_pred

layer_output = model.layers[-3].output
intermediate_model = Model(inputs=model.input,outputs=layer_output)
intermediate_prediction = intermediate_model.predict(X, batch_size = 64)
tmp = np.matrix(intermediate_prediction)
df = pandas.concat([df, pandas.DataFrame(tmp)], axis=1)

def KNN(filename, label):
   
    te = df[df['pred'] == label]
   
    X_test = te.drop(columns=['sim_number', 'pred']).values

    loaded_model = pickle.load(open(filename, 'rb'))
    pred = loaded_model.predict(X_test)
    
    return dict(zip(te['sim_number'].values, pred))

# predict_label_5 = KNN('knn_5.sav', 5)
# predict_label_6 = KNN('knn_6.sav', 6)
predict_label_5 = KNN('knn_default_5.sav', 5)
predict_label_6 = KNN('knn_default_6.sav', 6)

final_predict = []
for index, row in df.iterrows():
    if row['pred'] == 0:
        final_predict.append(450000)
    elif row['pred'] == 1:
        final_predict.append(500000)
    elif row['pred'] == 2:
        final_predict.append(1000000)
    elif row['pred'] == 3:
        final_predict.append(3000000)
    elif row['pred'] == 4:
        final_predict.append(5000000)
    elif row['pred'] == 5:
        final_predict.append(predict_label_5[row['sim_number']])
    else:
        final_predict.append(predict_label_6[row['sim_number']])
final_predict = [round(item) for item in final_predict]
df1 = df1.drop('pred', axis=1)
df1['price_vnd'] = final_predict
first_column = df1.pop('price_vnd')
df1.insert(0, 'price_vnd', first_column)

df1.to_csv('CNN+LSTM+KNN.csv', index = False)