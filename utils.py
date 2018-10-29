import random
import re
import sys
import time
import os

import numpy as np
from scipy import stats, spatial
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

from options import load_arguments




def load_sent(path, max_size=-1):
    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            tmp = line.rstrip('\n\r').split(' ||| ')
            #print(tmp)
            tmp0 = []
            tmp1 = []
            tmp0 = tmp[0].split(' ')
            tmp1 = tmp[1].split(' ')
            #print(tmp0,tmp1)
            if len(tmp0) != len(tmp1):
                print(tmp0, tmp1)
            try: 
                for i in range(len(tmp0)):
                    data.append([tmp0[i],tmp1[i]])
            except:
                print(tmp0,tmp1)
            data.append([])
            #break
    return data



def load_embedding(emb_file):
    data = []
    embedding2id = {'<pad>':0,'<s>':1}
    # word2id = {}
    id2embedding = ['<pad>','<s>']
    embedding_old = {}
    # id2word = []
    with open(emb_file) as f:
        for line in f:
            try:
                parts = line.split()
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]])
                embedding2id[word] = len(embedding2id)
                id2embedding.append(word)
                embedding_old[word] = vec
            except:
                print(len(embedding2id))
    return embedding2id,id2embedding, embedding_old




def convert_tag_to_id(X):
    tag2id = {'<pad>':0,'<s>':1,'</s>':2}
    id2tag = ['<pad>','<s>','</s>']
    for x in X:
        if len(x) != 0:
            try:
                tmp = x[1]
            except:
                print(x)
            if tmp not in tag2id:
                tag2id[tmp] = len(tag2id)
                id2tag.append(tmp)
    return (tag2id, id2tag)



def preprocess_data_according_to_rules(data, embedding_old):
    for i in range(len(data)):
        if len(data[i]) > 0:
            tmp = data[i][0]
#             def is_number(s):
#                 try:
#                     float(s)
#                     return True
#                 except ValueError:
#                     return False
#             if is_number(tmp):
#                 data[i][0] = 'num'
            if data[i][0] not in embedding_old:
                data[i][0] = 'UUUNKKK'
    return data



def construct_word_id(data):
    word2id = {'<pad>':0,'<s>':1,'</s>':2,'UUUNKKK':3}
    id2word = ['<pad>','<s>','</s>','UUUNKKK']
    for i in range(len(data)):
        if len(data[i]) > 0:
            tmp = data[i][0]
            if tmp not in word2id:
                word2id[tmp] = len(word2id)
                id2word.append(tmp)
    return word2id, id2word



def construct_embedding(embedding_size, dim_emb, embedding_old, word2id):
    #embedding = np.random.random_sample((embedding_size, dim_emb)) - 0.5
    embedding = np.zeros((embedding_size, dim_emb))
    for word in word2id:
        try: 
            embedding[word2id[word]] = embedding_old[word]
        except:
            print(word)
    return embedding




# data preprocessing and batch generation 

# get a list of x and y
# and turn letters into lower case
# turn unknown word to UUUNKKK
# add start and end token
def turn_data_into_x_y(dataset, word2id):
    x = []
    y = []
    x_tmp = ['<s>']
    y_tmp = []
    for i in range(len(dataset)):
        pair = dataset[i]
        if len(pair) == 2:
            word = pair[0] #.lower()
            if word not in word2id:
                word = 'UUUNKKK'
                
#             def is_number(s):
#                 try:
#                     float(s)
#                     return True
#                 except ValueError:
#                     return False
                                    
#             if is_number(word):
#                 word = 'num'
            
            x_tmp.append(word)
            y_tmp.append(pair[1])
        elif len(pair) == 0:
            x_tmp.append('</s>')
            x.append(x_tmp)
            y.append(y_tmp)
            x_tmp = ['<s>']
            y_tmp = []
        else:
            print("error at index", i)
    return x, y










def get_batch(x, y, word2id, tag2id):
    pad = word2id['<pad>']
    pad_tag = tag2id['<pad>']
    inputs_x, outputs_y, weights = [], [], []
    max_len = max([len(sent) for sent in x])
    for i in range(len(x)):
        line = x[i] # sentence with start and end symbols
        l = len(line) # sentence length
        padding = [pad] * (max_len - l)
        padding_tag = [pad_tag] * (max_len - l)
        tmp_line_x, tmp_line_y = [], []
        for word in line:
            tmp_line_x.append(word2id[word])
        for tag in y[i]:
            tmp_line_y.append(tag2id[tag])
        inputs_x.append(tmp_line_x+padding)
        outputs_y.append([tag2id['<s>']]+tmp_line_y+[tag2id['</s>']]+padding_tag)
        weights.append([1.0] * (l+1-1) + [0.0] * (max_len-l))
        
        
    return {'enc_inputs': inputs_x,
            'targets': outputs_y,
            'batch_size': len(inputs_x),
            'weights': weights,
            'len': max_len,
            'size': len(x)}


def get_batches(x, y, word2id, tag2id, batch_size):
    n = len(x)
    order = range(n)
    z = sorted(zip(order, x, y), key=lambda i: len(i[1]))
    order, x, y = zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(x[s:t], y[s:t], word2id, tag2id))
        s = t

    return batches, order














# BiLSTM



# BiLSTM


def create_cell(dim, dropout):
    cell = tf.nn.rnn_cell.LSTMCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    return cell

def retrive_var(scopes):
    var = []
    for scope in scopes:
        var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)
    return var

def create_model(sess, dim_h, n_tag, load_model=False, model_path=''):
    model = Model(dim_h, n_tag)
    if load_model:
        print('Loading model from ...')
        model.saver.restore(sess, model_path)
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    
    return model

def feed_dictionary(model, batch, dropout, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.targets: batch['targets'],
                 model.weights: batch['weights']}
    return feed_dict




# SPEN


# spen infnet + tlm


def get_batch(x, y, word2id, tag2id):
    pad = word2id['<pad>']
    pad_tag = tag2id['<pad>']
    inputs_x, outputs_y, tlm_outputs_y, weights, tlm_weights = [], [], [], [], []
    next_inputs_x = []
    
    inputs_x_reverse, outputs_y_reverse, tlm_outputs_y_reverse = [], [], []
    next_inputs_x_reverse = []
    
    max_len = max([len(sent) for sent in x])
    for i in range(len(x)):
        line = x[i] # sentence with start and end symbols
        l = len(line) # sentence length
        padding = [pad] * (max_len - l)
        padding_plus_one = [pad] * (max_len - l + 1)
        padding_tag = [pad_tag] * (max_len - l)
        padding_tag_plus_one = [pad_tag] * (max_len - l + 1)
        tmp_line_x, tmp_line_y = [], []
        for word in line:
            tmp_line_x.append(word2id[word])
        for tag in y[i]:
            tmp_line_y.append(tag2id[tag])
            
        inputs_x.append(tmp_line_x+padding)
        inputs_x_reverse.append(tmp_line_x[::-1]+padding)
        
        next_inputs_x.append(tmp_line_x[1:]+padding_plus_one)
        next_inputs_x_reverse.append(tmp_line_x[::-1][1:]+padding_plus_one)
        
        outputs_y.append([tag2id['<s>']]+tmp_line_y+[tag2id['</s>']]+padding_tag)
        outputs_y_reverse.append([tag2id['</s>']]+tmp_line_y[::-1]+[tag2id['<s>']]+padding_tag)
        
        tlm_outputs_y.append(tmp_line_y+[tag2id['</s>']]+padding_tag_plus_one)
        tlm_outputs_y_reverse.append(tmp_line_y[::-1]+[tag2id['<s>']]+padding_tag_plus_one)
        
        weights.append([1.0] * (l+1-1) + [0.0] * (max_len-l))
        tlm_weights.append([1.0] * (l-1) + [0.0] * (max_len-l+1))
        
        # weights_reverse and tlm_weights_reverse are the same as weights and tlm_weights
        
        
    return {'enc_inputs': inputs_x,
            'enc_inputs_reverse': inputs_x_reverse,
            'next_enc_inputs': next_inputs_x,
            'next_enc_inputs_reverse': next_inputs_x_reverse,
            'targets': outputs_y,
            'targets_reverse': outputs_y_reverse,
            'tlm_targets': tlm_outputs_y,
            'tlm_targets_reverse': tlm_outputs_y_reverse,
            'batch_size': len(inputs_x),
            'weights': weights,
            'tlm_weights': tlm_weights,
            'len': max_len,
            'size': len(x)}

def get_batches(x, y, word2id, tag2id, batch_size):
    n = len(x)
    order = range(n)
    z = sorted(zip(order, x, y), key=lambda i: len(i[1]))
    order, x, y = zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        if (s-t) < batch_size:
            s = t-batch_size
        batches.append(get_batch(x[s:t], y[s:t], word2id, tag2id))
        s = t

    return batches, order




def feed_dictionary(model, batch, dropout, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.enc_inputs_reverse: batch['enc_inputs_reverse'],
                 model.next_enc_inputs: batch['next_enc_inputs'],
                 model.next_enc_inputs_reverse: batch['next_enc_inputs_reverse'],
                 model.targets: batch['targets'],
                 model.targets_reverse: batch['targets_reverse'],
                 model.tlm_targets: batch['tlm_targets'],
                 model.tlm_targets_reverse: batch['tlm_targets_reverse'],
                 model.weights: batch['weights'],
                 model.tlm_weights: batch['tlm_weights']}
    return feed_dict





def create_cell_gru(dim, dropout):
    cell = tf.nn.rnn_cell.GRUCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    return cell


















