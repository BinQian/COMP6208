MAX_LEN = 105
import os
import pickle
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from time import time
from keras.callbacks import TensorBoard
from sklearn.model_selection import KFold


import datetime

def model_conv1D(emb_matrix):

    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=105,
        trainable=False
    )

    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(105,))
    seq2 = Input(shape=(105,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])


    # Add the magic features
    magic_input = Input(shape=(4,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    # Merge the Magic and distance features with the difference layer
    merge = concatenate([diff, mul,magic_dense])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2, magic_input], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


global_path= os.path.dirname(__file__)


stacking_test_preds={}  # dict for 5 test_set predictions 
stacking_output={}    # dict for features in the stacking model


model_conv1D = model_conv1D(embeddings)

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
    index+=1
    RawX_train=fivefolds_Xtrain[index]
    RawY_train=fivefolds_Ytrain[index]
    RawY_train.index= range(len(RawY_train))
    X_test=fivefolds_Xtest[index]
    Y_test=fivefolds_Ytest[index]
    magic_features_train[index].index = range(len(magic_features_train[index]))
    skf = KFold(n_splits=10,shuffle=True)
    for train_index,val_index in skf.split(RawX_train['left'], RawY_train):
        X_train={'left':RawX_train['left'][train_index],'right':RawX_train['right'][train_index]}
        Y_train=RawY_train[train_index]
        X_validation={'left':RawX_train['left'][val_index],'right':RawX_train['right'][val_index]}
        Y_validation=RawY_train[val_index]
        magic_train=magic_features_train[index].loc[train_index]
        magic_val=magic_features_train[index].loc[val_index]
        break
    
    # X_test = pickle.load(open(global_path+"/cache/nonStemX_test.mat","rb"))            
    weight_val = np.ones(len(Y_validation))
    weight_val *= 0.472001959
    weight_val[Y_validation==0] = 1.309028344
    
    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = global_path+'/base_models/1DCNN_model_magic.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    tbCallBack = TensorBoard(
                            log_dir=global_path+'/base_model_log',
                            histogram_freq=0,
                            write_grads=True,
                            write_images=True)
    
    tbCallBack.set_model(model_conv1D)
    # Start training
    training_start_time = time()
    
    malstm_trained = model_conv1D.fit([X_train['left'], X_train['right'],magic_train], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                                validation_data=([X_validation['left'], X_validation['right'],magic_val], Y_validation,weight_val),class_weight=class_weight,callbacks=[early_stopping,model_checkpoint,tbCallBack])
    
    model_conv1D.load_weights(bst_model_path)
    stacking_preds=model_conv1D.predict([X_test['left'],X_test['right'],magic_features_test[index]], batch_size=128, verbose=1)
    stacking_output[index]=stacking_preds
    real_predicts=model_conv1D.predict([allX_test['left'],allX_test['right'],all_magic], batch_size=8192, verbose=1)
    stacking_test_preds[index]=real_predicts
    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

stacking_features_X=np.concatenate((stacking_output[1],stacking_output[2],stacking_output[3],stacking_output[4],stacking_output[5])) #shape=(len(training_set,1))
stacking_features_Y=np.concatenate((fivefolds_Ytest[1],fivefolds_Ytest[2],fivefolds_Ytest[3],fivefolds_Ytest[4],fivefolds_Ytest[5]))#shape=(len(training_set,1))
real_output_preds=(stacking_test_preds[1]+stacking_test_preds[2]+stacking_test_preds[3]+stacking_test_preds[4]+stacking_test_preds[5])/5 #shape=(len(test_set,1))  when convert to cvs, dont forget to .ravel()



f = open(global_path+"/ensemble_data/1DCNN_stacking_features_X.mat", "wb")
pickle.dump(stacking_features_X, f)
f.close()

f = open(global_path+"/ensemble_data/1DCNN_stacking_features_Y.mat", "wb")
pickle.dump(stacking_features_Y, f)
f.close()

f = open(global_path+"/ensemble_data/1DCNN_real_output_preds.mat", "wb")
pickle.dump(real_output_preds, f)
f.close()
