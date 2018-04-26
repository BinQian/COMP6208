import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def generate_stem(x):

    """


    """
    snowballStemmer = SnowballStemmer('english')
    x = text.re.sub("[^a-zA-Z0-9]"," ", x) # match all alphabetic and number, replace others with " "
    x = (" ").join([snowballStemmer.stem(z) for z in x.split(" ")]) #stmmer works for single words, not sentences
    x = " ".join(x.split())  # in case there is blank at the end of the sentence
    return x


global_path = os.path.dirname(__file__)
#
train = pd.read_csv(global_path + '/rawdata/train.csv')
test = pd.read_csv(global_path + '/rawdata/test.csv')
print("start")

#lower() is original python function
train['question1_stem'] = train['question1'].astype(str).apply(lambda x:generate_stem(x.lower()))
train['question2_stem'] = train['question2'].astype(str).apply(lambda x:generate_stem(x.lower()))
test['question1_stem'] = test['question1'].astype(str).apply(lambda x:generate_stem(x.lower()))
test['question2_stem'] = test['question2'].astype(str).apply(lambda x:generate_stem(x.lower()))

train.to_csv(global_path + '/rawdata/train_stemmer.csv')
test.to_csv(global_path + '/rawdata/test_stemmer.csv')

feats = ["question1_stem","question2_stem"]
train = pd.read_csv(global_path + '/rawdata/train_stemmer.csv',encoding = 'ISO-8859-1')[feats]
test = pd.read_csv(global_path + '/rawdata/test_stemmer.csv',encoding = 'ISO-8859-1')[feats]
data_all = pd.concat([train,test])

#preserving local ordering info, extract 2-grams of words in addition to the 1-grams (individual words)
ngram_range = (1,2)
#When building the vocabulary, ignore terms that have a document frequency lower than min_df
min_df = 3
vect = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
print("TfidfVectorizer")
corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus += data_all[f].values.tolist()

vect_orig = vect.fit(corpus)

for f in feats:
    tfidfMatrix = vect_orig.transform(data_all[f].values.tolist())
    train_tfidf = tfidfMatrix[:train.shape[0]]
    test_tfidf = tfidfMatrix[train.shape[0]:]
    pd.to_pickle(train_tfidf, global_path + '/cache/train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf, global_path + '/cache/test_%s_tfidf.pkl'%f)

print("save")
train_question1_stem_tfidf = pd.read_pickle(global_path + '/cache/train_question1_stem_tfidf.pkl')[:]
train_question2_stem_tfidf = pd.read_pickle(global_path + '/cache/train_question2_stem_tfidf.pkl')[:]
test_question1_stem_tfidf = pd.read_pickle(global_path + '/cache/test_question1_stem_tfidf.pkl')[:]
test_question2_stem_tfidf = pd.read_pickle(global_path + '/cache/test_question2_stem_tfidf.pkl')[:]

train_length = train_question1_stem_tfidf.shape[0]
train_cosine_similarities = np.zeros([1, train_length])

test_length = test_question1_stem_tfidf.shape[0]
test_cosine_similarities = np.zeros([1, test_length])
print(test_length)

for i in range(0,train_length):
    train_cosine_similarities[0][i] = linear_kernel(train_question1_stem_tfidf[i], train_question2_stem_tfidf[i])[0]

for i in range(0,test_length):
    test_cosine_similarities[0][i] = linear_kernel(test_question1_stem_tfidf[i], test_question2_stem_tfidf[i])[0]

print("finally")
pd.to_pickle(train_cosine_similarities, global_path + '/cache/train_cosine_similarities.pkl')
pd.to_pickle(test_cosine_similarities, global_path + '/cache/test_cosine_similarities.pkl')