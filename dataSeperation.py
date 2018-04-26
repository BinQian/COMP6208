import os
import pickle
import itertools
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



validation_size = 40000
global_path= os.path.dirname(__file__)

print("global_path:"+global_path)

train_df = pickle.load(open(global_path+"/cache/nonStemtrainMy.dat","rb"))
magic_train = pickle.load(open(global_path+"/cache/train_magic.pkl","rb"))

questions_cols = ['question1', 'question2']
X = train_df[questions_cols]
Y = train_df['is_duplicate']

max_seq_length = 105

X_train, X_validation, Y_train, Y_validation, train_magic, validation_magic = train_test_split(X, Y, magic_train, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
# X_test = {'left': test_df.question1, 'right': test_df.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

f = open(global_path+"/cache2/nonStemX_train.mat", "wb")
pickle.dump(X_train, f)
f.close()

f = open(global_path+"/cache2/nonStemX_validation.mat", "wb")
pickle.dump(X_validation, f)
f.close()

f = open(global_path+"/cache2/nonStemY_train.mat", "wb")
pickle.dump(Y_train, f)
f.close()

f = open(global_path+"/cache2/nonStemY_validation.mat", "wb")
pickle.dump(Y_validation, f)
f.close()

f = open(global_path+"/cache2/nonStemMagic_train.mat", "wb")
pickle.dump(train_magic, f)
f.close()

f = open(global_path+"/cache2/nonStemMagic_validation.mat", "wb")
pickle.dump(validation_magic, f)
f.close()
