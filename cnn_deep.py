# Sentiment analysis of collection of 100000 reviews of IMDB
# Goal: prediction of user's opinion based on her comments


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb


max_features = 25000
maxlen = 300
batch_size = 32
embedding_dims = 100
nb_filters = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 6


X_train=train_vec
X_test=test_vec
y_train=train_label
y_test=test_label

print('Build model')
model = Sequential()

model.add(Embedding(max_features, embedding_dims))
model.add(Dropout(0.25))

model.add(Convolution1D(input_dim=embedding_dims,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())

output_size = nb_filters * (((maxlen - filter_length) / 1) + 1) / 2
model.add(Dense(output_size, hidden_dims))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(hidden_dims, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, validation_data=(X_test, y_test))
