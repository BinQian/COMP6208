from sklearn.model_selection import StratifiedKFold
import pickle
from keras.preprocessing.sequence import pad_sequences
import itertools
import pandas as pd
global_path = 'D:\\Kaggle\\stacking\\data'
train=pickle.load(open(global_path+"\\cache/nonStemtrainMy.dat","rb"))
test=pickle.load(open(global_path+"\\cache/nonStemtestMy.dat","rb"))

# zero padding the test set
test_bi={'left': test['question1'],'right':test['question2']}
for dataset, side in itertools.product([test_bi], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=105)

X={'left': train['question1'],'right':train['question2']}
Y=train['is_duplicate']

skf = StratifiedKFold(n_splits=5)

index=1
new_X_train={}
new_Y_train={}
new_X_one={}
new_Y_one={}
new_magic_train={}
new_magic_test={}
magic_features=pd.read_pickle('D:\Kaggle\stacking/train_magic.pkl')
for train_index, test_index in skf.split(X['left'], Y):
    temp_X_train={'left':X['left'][train_index],'right':X['right'][train_index]}
    temp_Y_train=Y[train_index]
    temp_X_one={'left':X['left'][test_index],'right':X['right'][test_index]}
    temp_Y_one=Y[test_index]
    temp_magic_train=magic_features.loc[train_index]
    temp_magic_test=magic_features.loc[test_index]
    new_X_train[index]=temp_X_train
    new_Y_train[index]=temp_Y_train
    new_X_one[index]=temp_X_one
    new_Y_one[index]=temp_Y_one
    new_magic_train[index]=temp_magic_train
    new_magic_test[index]=temp_magic_test
    index+=1

f=open(global_path+"/fivefolds_Xtrain.mat", "wb")
pickle.dump(new_X_train, f)
f.close()
f=open(global_path+"/fivefolds_Ytrain.mat", "wb")
pickle.dump(new_Y_train, f)
f.close()
f=open(global_path+"/fivefolds_Xtest.mat", "wb")
pickle.dump(new_X_one, f)
f.close()
f=open(global_path+"/fivefolds_Ytest.mat", "wb")
pickle.dump(new_Y_one, f)
f.close()
f=open(global_path+"/new_magic_train.pkl", "wb")
pickle.dump(new_magic_train, f)
f.close()

f=open(global_path+"/new_magic_test.pkl", "wb")
pickle.dump(new_magic_test, f)
f.close()

f = open(global_path+"/nonStemXtestMy_105.mat", "wb")
pickle.dump(test_bi, f)
f.close()