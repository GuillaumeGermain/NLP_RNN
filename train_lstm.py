#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gc, os, re
import tensorflow as tf
import sys, math, datetime 
from random import shuffle

#MAIN_DIR = '/root/kaggle/restaurantAvis/cache/' #Alain
MAIN_DIR = '' #Guillaume


BEST_TRAIN_ACCURACY_MODEL = MAIN_DIR + 'train_model.ckpt'
BEST_DEV_ACCURACY_MODEL = MAIN_DIR + 'dev_model.ckpt'

# Load Data
MOTS_UTILES = MAIN_DIR + 'mots_utiles.txt'
MOTS_NECESSAIRES = MAIN_DIR + 'mots_necessaires.txt'
X_TRAIN_CSV = MAIN_DIR + 'X_train.csv'
Y_TRAIN_CSV = MAIN_DIR + 'y_train.csv'
X_DEV_CSV = MAIN_DIR + 'X_dev.csv'
Y_DEV_CSV = MAIN_DIR + 'y_dev.csv'
PRETRAINED_VOCAB_CSV = MAIN_DIR + 'pretrained_vocab.csv'
TO_TRAIN_VOCAB_CSV = MAIN_DIR + 'to_train_vocab.csv'

EMB_DIM = 50 # 50,100, or 300
EMBS_CSV = MAIN_DIR + 'pretrained_embs_' + str(EMB_DIM) + '.csv'

pretrained_vocab = np.loadtxt(PRETRAINED_VOCAB_CSV, delimiter=';',dtype ='str')
only_in_train = np.loadtxt(TO_TRAIN_VOCAB_CSV, delimiter=';',dtype ='str')
pretrained_embs = np.loadtxt(EMBS_CSV, delimiter=';')

vocab = np.concatenate((pretrained_vocab, only_in_train))
#print(len(vocab))
#print(vocab[39009])


# Load the dataframes
X_train = pd.read_csv(X_TRAIN_CSV)
y_train = pd.read_csv(Y_TRAIN_CSV,names=['i','v'])
X_dev = pd.read_csv(X_DEV_CSV)
y_dev = pd.read_csv(Y_DEV_CSV,names=['i','v'])

regularizer_scale= 0.1
dropout_keep_prob = 0.8
learning_rate = 0.0001


REVIEW_LENGTH = 389

tf.reset_default_graph()
regularizer = tf.contrib.layers.l2_regularizer(scale=regularizer_scale)

vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
    mapping=tf.constant(vocab),
    default_value=len(vocab),
    dtype=tf.string)

pretrained_embs_tf = tf.get_variable(
    name="embs_pretrained",
    initializer=tf.constant_initializer(pretrained_embs, dtype=tf.float32),
    shape=pretrained_embs.shape,
    trainable=False)

train_embeddings = tf.get_variable(
    name="embs_only_in_train",
    shape=[len(only_in_train), EMB_DIM],
    initializer=tf.random_uniform_initializer(-0.04, 0.04),
    regularizer=regularizer,
    trainable=True)

unk_embedding = tf.get_variable(
    name="unk_embedding",
    shape=[1, EMB_DIM],
    initializer=tf.random_uniform_initializer(-0.04, 0.04),
    regularizer=regularizer,
    trainable=True)

embeddings = tf.concat([pretrained_embs_tf, train_embeddings, unk_embedding], axis=0)

X_t_review = tf.placeholder(shape = [ None, REVIEW_LENGTH],dtype = tf.string, name = "X_t_review")
y_t = tf.placeholder(shape = [ None, 1 ], dtype = tf.float32, name = "y_t")
# y_t_2 = tf.one_hot(y_t, 6)
phase = tf.placeholder(tf.bool, name='phase')

def lstm_to_train(string_tensor, scope):
    with tf.variable_scope(scope):
        string_tensor_num = vocab_lookup.lookup(string_tensor)
        string_tensor_embeded = tf.nn.embedding_lookup(embeddings, string_tensor_num)
        cell = tf.contrib.rnn.BasicLSTMCell(EMB_DIM)
        word_list = tf.unstack(string_tensor_embeded, axis=1)
        outputs, _ = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
        return outputs[-1]



lstm_review = lstm_to_train(X_t_review, 'review')


# add sigmoid
def dense(x, size, scope):
        h1 = tf.contrib.layers.fully_connected(x, size,activation_fn=None,weights_regularizer=regularizer,scope=scope)
        return h1
#        return  tf.nn.sigmoid(h1, 'sigmoid')
  
def dense_batch_relu(x, size, phase, scope):
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(
               x,
               size,
               activation_fn=None,
               weights_regularizer=regularizer,
               scope='dense')
            h2 = tf.contrib.layers.batch_norm(
                h1, 
                center=True, 
                scale=True, 
                is_training=phase,
                scope='bn')
            return tf.nn.relu(h2, 'relu')
            
def f2(net):
    return net

net = dense_batch_relu(lstm_review, 150, phase, 'layer1')
net = tf.cond(phase, lambda: tf.nn.dropout(net, dropout_keep_prob), lambda: f2(net))
net = dense_batch_relu(net, 100, phase, 'layer1b')
net = tf.cond(phase, lambda: tf.nn.dropout(net, dropout_keep_prob), lambda: f2(net))
net = dense_batch_relu(net, 50, phase, 'layer2')
preds = dense(net, 1, 'logits')
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.reduce_sum(reg_variables)
# cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_t_2))
# Define loss and optimizer
cost = tf.reduce_mean(tf.square(preds - y_t))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
loss = cost + reg_term

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.abs(preds - y_t))
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_t_2, 1), tf.argmax(logits, 1)),'float32'))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
saver = tf.train.Saver()

mini_batch_size = 512
num_epochs = 1000

m = len(X_train)
num_complete_minibatches = int(math.ceil(m/mini_batch_size))
if (m == (mini_batch_size * num_complete_minibatches)):
	mini_batches_list = list(range(num_complete_minibatches))
else:
	mini_batches_list = list(range(num_complete_minibatches))

m_dev = len(X_dev)
num_complete_minibatches_dev = int(math.ceil(m_dev/mini_batch_size))
if (m_dev == (mini_batch_size * num_complete_minibatches_dev)):
	mini_batches_list_dev = list(range(num_complete_minibatches_dev))
else:
	mini_batches_list_dev = list(range(num_complete_minibatches_dev))

def prepare_mini_batch(X,y,k): 
    first = k * mini_batch_size
    last = min((k + 1) * mini_batch_size, len(X))
    X_b = X[first : last]
    b_size = last - first;
    reviews = X_b['Review'].str.split()
    X_review = np.reshape(np.hstack(reviews),(b_size,REVIEW_LENGTH))
    y_b = y[first : last]
    y_b = np.asarray(y_b['v'].as_matrix(), dtype=np.float32)
    y_b = np.reshape(np.hstack(y_b),(b_size,1))
    return X_review,y_b
    
init_op = tf.global_variables_initializer()
init_tab = tf.tables_initializer()


with tf.Session() as sess:
    sess.run([init_op, init_tab])
    if os.path.isfile(BEST_TRAIN_ACCURACY_MODEL + '.index'):
        print ("parameters loaded")
        saver.restore(sess, BEST_TRAIN_ACCURACY_MODEL)
        
    best_train_accuracy = 0
    best_dev_accuracy = 0
    
    for epoch in range(num_epochs):
        epoch_cost = 0
        train_accuracy = 0
        shuffle(mini_batches_list)
        for k in mini_batches_list:
            X_review_b,y_b = prepare_mini_batch(X_train,y_train,k)
            minibatch_cost, minibatch_accuracy, _ = sess.run(
                    [ loss, accuracy, optimizer], 
                    feed_dict={
                        X_t_review :X_review_b,
                        y_t: y_b,
                        phase:True}
                )
            epoch_cost += minibatch_cost * y_b.shape[0]
            train_accuracy += minibatch_accuracy * y_b.shape[0]
        epoch_cost = epoch_cost / m
        train_accuracy = train_accuracy / m
        print("Cost mean epoch %i: %f" % (epoch, epoch_cost))
        print("Train Accuracy:", train_accuracy )
        if best_train_accuracy < train_accuracy:
            best_train_accuracy = train_accuracy
            saver.save(sess, BEST_TRAIN_ACCURACY_MODEL)
        if (epoch % 1 == 0):
            dev_accuracy = 0.0
            for k in mini_batches_list_dev:
                X_review_b,y_b = prepare_mini_batch(X_dev,y_dev,k)
                b_acc = sess.run([accuracy], feed_dict={
                    X_t_review:X_review_b,
                    y_t: y_b,
                    phase:False})
                dev_accuracy +=  b_acc[0] * y_b.shape[0]
            dev_accuracy = dev_accuracy / m_dev
            if best_dev_accuracy < dev_accuracy:
                best_dev_accuracy = dev_accuracy
                saver.save(sess, BEST_DEV_ACCURACY_MODEL)
            print("Dev Accuracy:", dev_accuracy )

        sys.stdout.flush()
