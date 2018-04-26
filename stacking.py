import pandas as pd
import os
import codecs
import csv
from sklearn.linear_model import LogisticRegression
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Nadam, Adam
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.model_selection import StratifiedKFold,KFold

global_path = os.path.dirname(__file__)



def average(test_a,test_b,test_c):
    #total length: 2345796
    return (test_a+test_b+test_c)/3

def logistic_regression(train_X,train_Y,test_X):
    # skf = StratifiedKFold(n_splits=10)
    # preds = np.zeros((len(test_X),2))
    model = LogisticRegression(C = 0.01,class_weight = {0: 1.309028344, 1: 0.472001959})

    # for train_index, test_index in skf.split(train_X, train_Y):
    #     model = model.fit(train_X[train_index], train_Y[train_index])
    #     preds  = preds + model.predict_proba(test_X)

    # preds = np.array([pred[1] for pred in preds])/10
    model = model.fit(train_X, train_Y)
    preds = model.predict_proba(test_X)
    preds = np.array([pred[1] for pred in preds])
    return preds

def xgboost_stacking(train_X,train_Y,test_X):
    # Classifier
    xgb = XGBClassifier()
    bst = xgb.fit(train_X, train_Y)
    preds = bst.predict_proba(test_X)
    preds = np.array([pred[1] for pred in preds])
    return preds

def decision_tree(train_X,train_Y,test_X):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_X, train_Y)
    preds = clf.predict_proba(test_X)
    return preds

def nn(train_X,train_Y,test_Y):
    skf = KFold(n_splits=10, shuffle=True)
    for train_index, val_index in skf.split(train_X,train_Y):
        X_train=train_X[train_index]
        Y_train=train_Y[train_index]
        X_validation=train_X[val_index]
        Y_validation=train_Y[val_index]
        break
    first=Input(shape=(3,))
    second=Dense(50,activation='relu')(first)
    second = Dense(25, activation='relu')(second)
    output=Dense(1,activation='sigmoid')(second)

    model=Model(inputs=first, outputs=output)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])
    weight_val = np.ones(len(Y_validation))
    weight_val *= 0.472001959
    weight_val[Y_validation == 0] = 1.309028344

    bst_model_path =global_path+'/ensemble_data/stacking_nn' + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    model=nn()
    model.fit(X_train, Y_train, batch_size=64, nb_epoch=10,
                              validation_data=(X_validation, Y_validation,weight_val),class_weight=class_weight,callbacks=[model_checkpoint])
    model.load_weights(bst_model_path)
    predicts=model.predict(test_Y, batch_size=128,
                                              verbose=1)
    return predicts

def concatenate(a,b,c):
    return np.concatenate((a, b, c),axis=1)


def convert_to_csv(preds,name):
    TEST_CSV = global_path+'/rawdata/test.csv'

    test_ids = []
    with codecs.open(TEST_CSV, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            test_ids.append(values[0])
    test_ids = np.array(test_ids)
    submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
    submission.to_csv(global_path+'/submissions/'+name+'.csv', index=False)

train_X_attention_ensemble = pd.read_pickle(global_path+'/ensemble_data/attention_stacking_features_X.mat') #404290,1
train_X_LSTM_ensemble = pd.read_pickle(global_path+'/ensemble_data/lstm_train_output.pkl') #404290,1
train_X_1DCNN_ensemble = pd.read_pickle(global_path+'/ensemble_data/1DCNN_stacking_features_X.mat') #404290,1

train_Y_ensemble = pd.read_pickle(global_path+'/ensemble_data/lstm_train_label.pkl') #404290,1

test_attention_ensemble = pd.read_pickle(global_path+'/ensemble_data/attention_real_output_preds.mat') # 2345796,1
test_1DCNN_ensemble = pd.read_pickle(global_path+'/ensemble_data/1DCNN_real_output_preds.mat') # 2345796,1
test_LSTM_ensemble = pd.read_pickle(global_path+'/ensemble_data/lstm_test_output.pkl') # 2345796,1

train_X_ensemble = concatenate(train_X_attention_ensemble,train_X_LSTM_ensemble,train_X_1DCNN_ensemble)
test_X_ensemble = concatenate(test_attention_ensemble,test_1DCNN_ensemble,test_LSTM_ensemble)



# average over three models
# convert_to_csv(average(test_attention_ensemble,test_1DCNN_ensemble,test_LSTM_ensemble),'average_model')

# logistic regression
convert_to_csv(logistic_regression(train_X_ensemble,train_Y_ensemble,test_X_ensemble),'logistic_regression_stacking_regular')

# Xgboost
# convert_to_csv(xgboost_stacking(train_X_ensemble,train_Y_ensemble,test_X_ensemble),'xgboost_stacking')

#decision treee
# convert_to_csv(logistic_regression(train_X_ensemble,train_Y_ensemble,test_X_ensemble))

#XGBOOST
# convert_to_csv(xgboost_stacking(train_X_ensemble,train_Y_ensemble,test_X_ensemble))

#nn
# nn(train_X_ensemble,train_Y_ensemble,test_X_ensemble)
