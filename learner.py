from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras import layers as N
from keras import losses
from time import sleep
import pymongo

client = pymongo.MongoClient('localhost', 27017)
db = client.bmexdata
trade_collection = db.candles


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


n = 420
price_scale_ratio = 10000
volume_scale_ratio = 1000000
hist = DataFrame(list(trade_collection.find().skip(trade_collection.count() - n)))
hist = hist.set_index('_id')
hist = hist.drop(['ot', 'ct'], axis=1)
hist['o'] = hist['o'] / price_scale_ratio
hist['h'] = hist['h'] / price_scale_ratio
hist['l'] = hist['l'] / price_scale_ratio
hist['c'] = hist['c'] / price_scale_ratio
hist['bv'] = hist['bv'] / volume_scale_ratio
hist['sv'] = hist['sv'] / volume_scale_ratio

# load dataset
dataset = hist
# print(dataset)
# sleep(10)
values = dataset.values
# print(values)
# # integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])

# ensure all data is float
values = values.astype('float32')

# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # print(scaled)

# specify the number of lag hours
input_sample = 120
output_shift = 60
n_features = 6
# frame as supervised learning
reframed = series_to_supervised(values, input_sample, output_shift)

# split into train and test sets
values = reframed.values
n_test = 30
n_train = reframed.shape[0] - n_test
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
target_col = -5
n_obs = input_sample * n_features
train_X, train_y = train[:, :n_obs], train[:, target_col]
test_X, test_y = test[:, :n_obs], test[:, target_col]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], input_sample, n_features))
test_X = test_X.reshape((test_X.shape[0], input_sample, n_features))
print('Train', train.shape)
print(train)
print('Train X', train_X.shape)
print(train_X)
print('Train y', train_y.shape)
print(train_y)
print('Test', test.shape)
print(test)
print('Test X', test_X.shape)
print(test_X)
print('Test y', test_y.shape)
print(test_y)
sleep(10)

# design network
neurons = 500
model = Sequential()
model.add(N.LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(N.LSTM(5, return_sequences=True))
model.add(N.LSTM(neurons, return_sequences=True))
model.add(N.LSTM(5))
model.add(N.Dense(1))
model.compile(loss=losses.logcosh, optimizer='adam')

# # load json and create model
# json_file = open('model-3x500-lstm-shift12.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights("model-3x500-lstm-shift12.h5")
# print("Loaded model from disk")
# model.compile(loss=losses.mean_squared_logarithmic_error, optimizer=opz)

# fit network
# history = model.fit(train_X, train_y, epochs=20, batch_size=30, validation_data=(test_X, test_y), verbose=1,
history = model.fit(train_X, train_y, epochs=15, batch_size=30, verbose=1, validation_data=(test_X, test_y),
                    shuffle=True)

# evaluate the model
# scores = model.evaluate(train_X, train_y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("ai_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("ai_model.h5")
print("Saved model to disk")

# print(history)
# plot history
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.legend()
# pyplot.show()

# make a prediction
pred = model.predict(test_X)
# print(type(yhat))
pred = pred.reshape(1, len(pred)).tolist()[0]
test_y = test_y.tolist()
print(pred)
print(test_y)

wins = 0
loses = 0
for i in range(len(pred)-1):
    if pred[i+1] > pred[i]:
        # print("Predict up")
        if test_y[i+1] > test_y[i]:
            # print('Win')
            wins += 1
        else:
            # print('Lose')
            loses += 1
    elif pred[i+1] < pred[i]:
        # print("Predict down")
        if test_y[i + 1] < test_y[i]:
            # print('Win')
            wins += 1
        else:
            # print('Lose')
            loses += 1
    else:
        print('Neutral')
print('Wins: %d, loses: %d. Win rate: %4f' % (wins, loses, wins/(wins+loses)))
with open("learn_test.log", "a") as myfile:
    myfile.write('Wins: %d, loses: %d. Win rate: %4f\n' % (wins, loses, wins/(wins+loses)))
# pyplot.plot(pred, label='predict')
# pyplot.plot(test_y, label='test')
# pyplot.legend()
# pyplot.show()
