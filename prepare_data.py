#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
#import gc, os,
#import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

#import math, datetime
#from random import shuffle
import nltk


#MAIN_DIR = '/root/kaggle/restaurantAvis/cache/' #Alain
MAIN_DIR = '' #Guillaume

TRAINEDMODEL = MAIN_DIR + 'model_multi_lstm1.ckpt'
VOCAB_ALL = MAIN_DIR + 'vocab_all.txt'

MOTS_UTILES = MAIN_DIR + 'mots_utiles.txt'
MOTS_NECESSAIRES = MAIN_DIR + 'mots_necessaires.txt'
DATASET_DIR = 'dataset/'

X_TRAIN_CSV = MAIN_DIR + 'X_train.csv'
Y_TRAIN_CSV = MAIN_DIR + 'y_train.csv'
X_DEV_CSV = MAIN_DIR + 'X_dev.csv'
Y_DEV_CSV = MAIN_DIR + 'y_dev.csv'

PRETRAINED_VOCAB_CSV = MAIN_DIR + 'pretrained_vocab.csv'
TO_TRAIN_VOCAB_CSV = MAIN_DIR + 'to_train_vocab.csv'
ONE_HOT_COLS_CSV = MAIN_DIR + 'one_hot_cols.csv'

EMBS_CSV = MAIN_DIR + 'pretrained_embs_'
GLOVE_DIR = MAIN_DIR + 'tools/embedding/glove.6B/'


# Load the data
train = pd.read_csv(DATASET_DIR + 'restaurant_train.tsv', delimiter='\t', quoting=3)
# dev = pd.read_csv(DATASET_DIR + 'restaurant_dev.tsv', delimiter='\t', quoting=3)
# train = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype=dtype, low_memory=True, encoding='utf-8')


# Quick look at what's in the dataframes
#print(train.count())
#train.head()
#print(dev.count())
#dev.head()


# Function to clean the text
def clean_text(data):
    words = nltk.word_tokenize(data.lower().replace('\\n',' ').replace('\\t',' ').replace('_',' '))
    return " ".join(words).replace('``','"').replace("n't","not")



# We save the whole vocabulary and the most used words
def save_list(filepath, aList):
    with open(filepath, 'w') as f:
        for l in aList:
            f.write(l + '\\n')
            
#### Preprocess data
def extract_features(df):
    df['CleanReview'] = df.apply(lambda row: clean_text(str(row['Review'])), axis=1)

extract_features(train)

# We re-create the review field

vectorizer = CountVectorizer(ngram_range=(1,1),stop_words=frozenset([]))
vectorizer.fit(train["CleanReview"])
mots_utiles = set(vectorizer.vocabulary_)
save_list(MOTS_UTILES, mots_utiles)
print(len(mots_utiles))


vectorizer = CountVectorizer(min_df=50, stop_words=frozenset([]))
vectorizer.fit(train["CleanReview"])
mots_necessaires = set(vectorizer.vocabulary_)
save_list(MOTS_NECESSAIRES, mots_necessaires)
print(len(mots_necessaires))

del vectorizer


# We estimate the optimal size for the RNN network
train['review_len'] = train['CleanReview'].apply(lambda x: len(x.split()))
print(max(train['review_len']))
print(train['review_len'].mean())
print(train['review_len'].quantile(q=0.90))


# We load the embedded words vectors
def load_pretrained_glove(dim):
    vocab = np.loadtxt(GLOVE_DIR + "glove.6B." + str(dim) + "d.txt",
                       delimiter = ' ',
                       dtype='str',
                       comments=None,
                       usecols=0)
    vectors = np.loadtxt(GLOVE_DIR + "glove.6B." + str(dim) + "d.txt",
                         delimiter = ' ',
                         comments=None,
                         usecols=(i+1 for i in range(dim)))
    return vocab, vectors


pretrained_vocab, pretrained_embs = load_pretrained_glove(50)
_, pretrained_embs_100 = load_pretrained_glove(100)
_, pretrained_embs_300 = load_pretrained_glove(300)
print(pretrained_vocab[:10])
print(pretrained_embs[:3,:])


# Remove the useless words to process the dataset (test and train)

# clean up unused words
mots_inutiles = [i for i in range(len(pretrained_vocab)) if pretrained_vocab[i] not in mots_utiles and len(pretrained_vocab[i]) > 1]
print(len(mots_inutiles))


pretrained_vocab = np.delete(pretrained_vocab, (mots_inutiles), axis=0)
pretrained_embs = np.delete(pretrained_embs, (mots_inutiles), axis=0)
pretrained_embs_100 = np.delete(pretrained_embs_100, (mots_inutiles), axis=0)
pretrained_embs_300 = np.delete(pretrained_embs_300, (mots_inutiles), axis=0)

print(pretrained_vocab.shape)
print(pretrained_embs.shape)
print(pretrained_embs_100.shape)


# We estimate the vocabulary known by the application

only_in_train = mots_necessaires - set(pretrained_vocab)
only_in_train = list(only_in_train)
only_in_train.append("<BLANK>")
vocab = list(pretrained_vocab) + only_in_train

print(len(only_in_train))
print(len(vocab))


REVIEW_LENGTH = 389
def preprocess_text(data,length):
        data = [w if w in vocab else '<UNK>' for w in str(data2).split()]
        if len(data) < length :
            data = data +['<BLANK>' for i in range(length - len(data))]
        return ' '.join(data[:length])


train['CleanReviewSized'] = train['CleanReview'].apply(lambda x:preprocess_text(x,REVIEW_LENGTH))


train.head(3)


X_train, X_dev, y_train, y_dev = train_test_split(train, train['Liked'], test_size=0.2)


X_train.to_csv(X_TRAIN_CSV)
y_train.to_csv(Y_TRAIN_CSV)
X_dev.to_csv(X_DEV_CSV)
y_dev.to_csv(Y_DEV_CSV)


np.savetxt(PRETRAINED_VOCAB_CSV, pretrained_vocab, delimiter=';',fmt='%s')
np.savetxt(TO_TRAIN_VOCAB_CSV, only_in_train, delimiter=';',fmt='%s')
np.savetxt(EMBS_CSV + '50.csv', pretrained_embs, delimiter=';')
np.savetxt(EMBS_CSV + '100.csv', pretrained_embs_100, delimiter=';')
np.savetxt(EMBS_CSV + '300.csv', pretrained_embs_300, delimiter=';')
