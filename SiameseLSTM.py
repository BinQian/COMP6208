# Model variables
from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
from keras.layers import Dense, concatenate
from keras.layers import Lambda, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
import os
global_path = os.path.dirname(__file__)

max_seq_length = 105
embedding_dim = 300

file_1=open(global_path+"/cache2/test_magic.pkl", "rb")
all_magic=pickle.load(file_1)
file_1.close()


file_1=open(global_path+"/cache2/fivefolds_Xtrain.mat", "rb")
fivefolds_Xtrain=pickle.load(file_1)
file_1.close()

file_1=open(global_path+"/cache2/fivefolds_Ytrain.mat", "rb")
fivefolds_Ytrain=pickle.load(file_1)
file_1.close()

file_1=open(global_path+"/cache2/fivefolds_Xtest.mat", "rb")
fivefolds_Xtest=pickle.load(file_1)
file_1.close()

file_1=open(global_path+"/cache2/fivefolds_Ytest.mat", "rb")
fivefolds_Ytest=pickle.load(file_1)
file_1.close()

file_1=open(global_path+"/cache2/nonStemembeddings.dat", "rb")
embeddings=pickle.load(file_1)
file_1.close()

file_1=open(global_path+"/cache2/nonStemXtestMy_105.mat", "rb")
allX_test=pickle.load(file_1)
file_1.close()


file_1=open(global_path+"/cache2/new_magic_train.mat", "rb")
magic_features_train=pickle.load(file_1)
file_1.close()

file_1=open(global_path+"/cache2/new_magic_test.mat", "rb")
magic_features_test=pickle.load(file_1)
file_1.close()

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25

stacking_test_preds = {}  # dict for 5 test_set predictions
stacking_output = {}  # dict for features in the stacking model
for index in range(5):
    index += 1
    RawX_train = fivefolds_Xtrain[index]
    RawY_train = fivefolds_Ytrain[index]
    X_test = fivefolds_Xtest[index]
    Y_test = fivefolds_Ytest[index]
    magic_features_train[index].index = range(len(magic_features_train[index]))
    RawY_train.index = range(len(RawY_train))
    skf = StratifiedKFold(n_splits=10)
    for train_index, val_index in skf.split(RawX_train['left'], RawY_train):
        X_train = {'left': RawX_train['left'][train_index], 'right': RawX_train['right'][train_index]}
        Y_train = RawY_train[train_index]
        X_validation = {'left': RawX_train['left'][val_index], 'right': RawX_train['right'][val_index]}
        Y_validation = RawY_train[val_index]
        magic_train = magic_features_train[index].loc[train_index]
        magic_val = magic_features_train[index].loc[val_index]
        break

    class_weight = {0: 1.309028344, 1: 0.472001959}
    weight_val = np.ones(len(Y_validation))
    weight_val *= 0.472001959
    weight_val[Y_validation == 0] = 1.309028344


    def exponent_neg_manhattan_distance(vects):
        ''' Helper function for the similarity estimate of the LSTMs outputs'''
        left, right = vects
        return K.abs(left - right)


    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 50)


    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length,
                                trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden, kernel_initializer='random_normal', recurrent_initializer='random_normal')

    left_output = shared_lstm(encoded_left)

    right_output = shared_lstm(encoded_right)

    # last_layer=Dense(1,input_dim=50,activation='sigmoid')
    # prob=np.linalg.norm(distance,ord=1)

    # magic features
    magic_input = Input(shape=(4,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Lambda(exponent_neg_manhattan_distance, output_shape=eucl_dist_output_shape)(
        [left_output, right_output])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=eucl_dist_output_shape)([left_output, right_output])

    malstm_distance = concatenate([malstm_distance, mul, magic_dense])
    malstm_distance = BatchNormalization()(malstm_distance)  # pay attention to batchnormalize

    hidden_layer = Dense(12, activation='relu')

    hidden_var = hidden_layer(malstm_distance)
    hidden_var = BatchNormalization()(hidden_var)
    output_layer = Dense(1, activation='sigmoid')
    prob = output_layer(hidden_var)

    # Pack it all up into a model
    malstm = Model([left_input, right_input, magic_input], [prob])

    # Adadelta optimizer, with gradient clipping by norm

    malstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = global_path+'/base_models/stacking_lstm' + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    # Start training
    training_start_time = time()

    malstm_trained = malstm.fit([X_train['left'], X_train['right'], magic_train], Y_train, batch_size=batch_size,
                                nb_epoch=n_epoch, validation_data=(
        [X_validation['left'], X_validation['right'], magic_val], Y_validation, weight_val),
                                class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
    malstm.load_weights(bst_model_path)
    stacking_preds = malstm.predict([X_test['left'], X_test['right'], magic_features_test[index]], batch_size=128,
                                    verbose=1)
    stacking_output[index] = stacking_preds
    real_predicts = malstm.predict([allX_test['left'], allX_test['right'], all_magic], batch_size=8192, verbose=1)
    stacking_test_preds[index] = real_predicts
    print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                            datetime.timedelta(seconds=time() - training_start_time)))

stacking_features_X = np.concatenate((stacking_output[1], stacking_output[2], stacking_output[3], stacking_output[4],
                                      stacking_output[5]))  # shape=(len(training_set,1))
stacking_features_Y = np.concatenate((fivefolds_Ytest[1], fivefolds_Ytest[2], fivefolds_Ytest[3], fivefolds_Ytest[4],
                                      fivefolds_Ytest[5]))  # shape=(len(training_set,1))
real_output_preds = (stacking_test_preds[1] + stacking_test_preds[2] + stacking_test_preds[3] + stacking_test_preds[4] +
                     stacking_test_preds[
                         5]) / 5  # shape=(len(test_set,1))  when convert to cvs, dont forget to .ravel()
import csv
stacking_features_X.to_csv(global_path+'/ensemble_data/lstm_train_output.csv', index=False)  #will be used to train the stacking model as one features
stacking_features_Y.to_csv(global_path+'/ensemble_data/lstm_train_label.csv', index=False)  #will be used to train the stacking model as the label
real_output_preds.to_csv(global_path+'/ensemble_data/lstm_test_output.csv', index=False)  #will be used to predict the probability as one feature in the stacking model