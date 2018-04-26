
MAX_LEN = 105
import datetime
import os
import pickle
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from time import time
from sklearn.model_selection import KFold

def create_pretrained_embedding(embeddings, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    pretrained_weights = embeddings
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    # LAMBDA: Wraps arbitrary expression as a Layer object.
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_embedding,
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    # Based on: https://arxiv.org/abs/1606.01933

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    # Add the magic features
    magic_input = Input(shape=(4,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)


    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding,
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            #                 Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        #             Dropout(rate=projection_dropout),
    ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        #         Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        #         Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep, magic_dense])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    #     dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    #     dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2,magic_input], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])
    return model




# global_path=os.path.abspath('.')
global_path = os.path.dirname(__file__)
stacking_test_preds={}  # dict for 5 test_set predictions
stacking_output={}    # dict for features in the stacking model

embeddings = pickle.load(open(global_path+"/cache2/nonStemembeddings.dat","rb"))
decomposable_attention_model = decomposable_attention(embeddings)




batch_size = 64
n_epoch = 25
class_weight = {0: 1.309028344, 1: 0.472001959}

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

for index in range(5):
    index += 1
    RawX_train = fivefolds_Xtrain[index]
    RawY_train = fivefolds_Ytrain[index]
    RawY_train.index = range(len(RawY_train))
    X_test = fivefolds_Xtest[index]
    Y_test = fivefolds_Ytest[index]
    magic_features_train[index].index = range(len(magic_features_train[index]))
    skf = KFold(n_splits=10, shuffle=True)
    for train_index, val_index in skf.split(RawX_train['left'], RawY_train):
        X_train = {'left': RawX_train['left'][train_index], 'right': RawX_train['right'][train_index]}
        Y_train = RawY_train[train_index]
        X_validation = {'left': RawX_train['left'][val_index], 'right': RawX_train['right'][val_index]}
        Y_validation = RawY_train[val_index]
        magic_train = magic_features_train[index].loc[train_index]
        magic_val = magic_features_train[index].loc[val_index]
        break


    weight_val = np.ones(len(Y_validation))
    weight_val *= 0.472001959
    weight_val[Y_validation == 0] = 1.309028344
    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = global_path+'/base_models/decomposable_attention_model.h5'
#save the model with the best parameters
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
# Start training
    training_start_time = time()


    tbCallBack = TensorBoard(log_dir=global_path)



    model_trained = decomposable_attention_model.fit([X_train['left'], X_train['right'],magic_train], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                          validation_data=([X_validation['left'], X_validation['right'],magic_val], Y_validation,weight_val),class_weight=class_weight,callbacks=[early_stopping, model_checkpoint,tbCallBack])

    decomposable_attention_model.load_weights(bst_model_path)
    stacking_preds = decomposable_attention_model.predict([X_test['left'], X_test['right'],magic_features_test[index]], batch_size=128,
                                          verbose=1)
    stacking_output[index] = stacking_preds
    real_predicts =decomposable_attention_model.predict([allX_test['left'], allX_test['right'],all_magic], batch_size=128, verbose=1)
    stacking_test_preds[index] = real_predicts
    print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                            datetime.timedelta(seconds=time() - training_start_time)))

stacking_features_X=np.concatenate((stacking_output[1],stacking_output[2],stacking_output[3],stacking_output[4],stacking_output[5])) #shape=(len(training_set,1))
stacking_features_Y=np.concatenate((fivefolds_Ytest[1],fivefolds_Ytest[2],fivefolds_Ytest[3],fivefolds_Ytest[4],fivefolds_Ytest[5]))#shape=(len(training_set,1))
real_output_preds=(stacking_test_preds[1]+stacking_test_preds[2]+stacking_test_preds[3]+stacking_test_preds[4]+stacking_test_preds[5])/5 #shape=(len(t

f = open(global_path+"/ensemble_data/attention_stacking_features_X.mat", "wb")
pickle.dump(stacking_features_X, f)
f.close()

f = open(global_path+"/ensemble_data/attention_stacking_features_Y.mat", "wb")
pickle.dump(stacking_features_Y, f)
f.close()

f = open(global_path+"/ensemble_data/attention_real_output_preds.mat", "wb")
pickle.dump(real_output_preds, f)
f.close()

#
#