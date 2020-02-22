import random
import re
import sys
import time
import os

import numpy as np
from scipy import stats, spatial
from sklearn.decomposition import TruncatedSVD
import tensorflow.compat.v1 as tf

from options import load_arguments






### load data

def load_sent(path, max_size=4):
    data = []
    with open(path) as f:
        for line in f:
            tmp = line.split()
            data.append(tmp)
    return data




def load_embedding(emb_file):
    data = []
    embedding2id = {'<pad>':0, '<s>':1, '</s>':2}
    # word2id = {}
    id2embedding = ['<pad>', '<s>', '</s>']
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





def convert_tag_to_id(X, which):
    tag2id = {'<pad>':0, '<s>':1, '</s>':2}
    id2tag = ['<pad>', '<s>', '</s>']
    for x in X:
        if len(x) != 0:
            try:
                tmp = x[which]
            except:
                print(x)
            if tmp not in tag2id:
                tag2id[tmp] = len(tag2id)
                id2tag.append(tmp)
    return (tag2id, id2tag)



def preprocess_data_according_to_rules(data):
    for i in range(len(data)):
        if len(data[i]) > 0:
            if not data[i][0].islower():
                data[i].append(1) # there is at least one char upper case
            else:
                data[i].append(0) # all lower case
            
            def is_number(s):
                tmp_tf = 0
                for c in s:
                    if '0' <= c <= '9':
                        tmp_tf = 1
                if tmp_tf:
                    for c in s:
                        if not ('0' <= c <= '9' or c == '-' or c == ',' or c == '.'):
                            return False
                    return True
                else:
                    return False
            
            if is_number(data[i][0]):
                data[i].append(1)
            else:
                data[i].append(0)
            
            tmp = data[i][0]
            
#             def is_number(s):
#                 try:
#                     float(s)
#                     return True
#                 except ValueError:
#                     return False

            if tmp.lower() not in embedding_old:
                data[i][0] = 'unk'
            else:
                data[i][0] = tmp
    return data





def construct_word_id():
    data = train_data+dev_data+test_data
    word2id = {'<pad>':0,'<s>':1,'</s>':2} # UUUNKKK replaced by unk
    id2word = ['<pad>','<s>','</s>']
    for i in range(len(data)):
        if len(data[i]) > 0:
            tmp = data[i][0].lower()
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
# turn unknown word to UUUNKKK
def turn_data_into_x_y(dataset, word2id):
    x, y, pos, chunk, case, num = [], [], [], [], [], []
    x_tmp, y_tmp, pos_tmp, chunk_tmp, case_tmp, num_tmp = ['<s>'], [], [], [], [], []
    for i in range(len(dataset)):
        tup = dataset[i]
        if len(tup) == 6:
            word = tup[0].lower() #.lower()
            if word not in word2id:
                print(word)
            x_tmp.append(tup[0])
            y_tmp.append(tup[3])
            pos_tmp.append(tup[1])
            chunk_tmp.append(tup[2])
            case_tmp.append(tup[4])
            num_tmp.append(tup[5])
        elif len(tup) == 0:
            x_tmp.append('</s>')
            x.append(x_tmp)
            y.append(y_tmp)
            pos.append(pos_tmp)
            chunk.append(chunk_tmp)
            case.append(case_tmp)
            num.append(num_tmp)
            x_tmp, y_tmp, pos_tmp, chunk_tmp, case_tmp, num_tmp = ['<s>'], [], [], [], [], []
        else:
            print("error at index", i)
    return x, y, pos, chunk, case, num






# all_chars = ['<padunk>']+list(string.punctuation+string.ascii_uppercase+string.ascii_lowercase+string.digits)
# id2char = all_chars
# char2id = {}
# for x in all_chars:
#     char2id[x] = all_chars.index(x)




def construct_embedding_char():
    embedding_char = np.zeros((len(id2char), 16))
    return embedding_char
    
# embedding_char_global = construct_embedding_char()








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

# def feed_dictionary(model, batch, dropout, learning_rate=None):
#     feed_dict = {model.dropout: dropout,
#                  model.learning_rate: learning_rate,
#                  model.batch_len: batch['len'],
#                  model.batch_size: batch['size'],
#                  model.enc_inputs: batch['enc_inputs'],
#                  model.targets: batch['targets'],
#                  model.weights: batch['weights']}
#     return feed_dict









### SPEN
### spen infnet + tlm


def get_batch(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id):
    pad = word2id['<pad>']
    pad_tag = tag2id['<pad>']
    inputs_x, outputs_y, tlm_outputs_y, weights, tlm_weights, tlm_targets_pos = [], [], [], [], [], []
    inputs_x_char = []
    next_inputs_x = []
    inputs_pos, inputs_chunk, inputs_case, inputs_num = [], [], [], []
    
    inputs_x_reverse, outputs_y_reverse, tlm_outputs_y_reverse, tlm_targets_pos_reverse = [], [], [], []
    next_inputs_x_reverse = []
    inputs_pos_reverse, inputs_chunk_reverse, inputs_case_reverse, inputs_num_reverse = [], [], [], []
    
    len_char = []
    
    max_len = max([len(sent) for sent in x])
    for i in range(len(x)):
        line = x[i] # sentence with start and end symbols
        l = len(line) # sentence length
        padding = [pad] * (max_len - l)
        padding_plus_one = [pad] * (max_len - l + 1)
        padding_tag = [pad_tag] * (max_len - l)
        padding_tag_plus_one = [pad_tag] * (max_len - l + 1)
        tmp_line_x, tmp_line_y, tmp_line_pos, tmp_line_chunk, tmp_line_case, tmp_line_num = [], [], [], [], [], []
        tmp_cline_x = []
        tmp_tmp_cline_x = []
        for word in line:
            tmp_line_x.append(word2id[word.lower()]) # lower!!
            for c in word:
                try:
                    tmp_tmp_cline_x.append(char2id[c])
                except:
                    print('char does not exist', c)
            tmp_cline_x.append(tmp_tmp_cline_x)
            tmp_tmp_cline_x = []
        for tag in y[i]:
            tmp_line_y.append(tag2id[tag])
        for pos_single in pos[i]:
            tmp_line_pos.append(pos2id[pos_single])
        for chunk_single in chunk[i]:
            tmp_line_chunk.append(chunk2id[chunk_single])
        for case_single in case[i]:
            tmp_line_case.append(case_single)
        for num_single in num[i]:
            tmp_line_num.append(num_single)
        
        inputs_x.append(tmp_line_x+padding)
        inputs_x_char.append(tmp_cline_x)
        len_char.append(len(tmp_cline_x))
        inputs_x_reverse.append(tmp_line_x[::-1]+padding)
        
        next_inputs_x.append(tmp_line_x[1:]+padding_plus_one)
        next_inputs_x_reverse.append(tmp_line_x[::-1][1:]+padding_plus_one)
        
        outputs_y.append([tag2id['<s>']]+tmp_line_y+[tag2id['</s>']]+padding_tag)
        outputs_y_reverse.append([tag2id['</s>']]+tmp_line_y[::-1]+[tag2id['<s>']]+padding_tag)
        
        inputs_pos.append([pos2id['<s>']]+tmp_line_pos+[pos2id['</s>']]+padding_tag)
        inputs_pos_reverse.append([pos2id['</s>']]+tmp_line_pos[::-1]+[pos2id['<s>']]+padding_tag)
        inputs_chunk.append([chunk2id['<s>']]+tmp_line_chunk+[chunk2id['</s>']]+padding_tag)
        inputs_chunk_reverse.append([chunk2id['</s>']]+tmp_line_chunk[::-1]+[chunk2id['<s>']]+padding_tag)
        
        inputs_case.append([0]+tmp_line_case+[0]+padding_tag)
        inputs_case_reverse.append([0]+tmp_line_case[::-1]+[0]+padding_tag)
        inputs_num.append([0]+tmp_line_num+[0]+padding_tag)
        inputs_num_reverse.append([0]+tmp_line_num[::-1]+[0]+padding_tag)
        
        tlm_outputs_y.append(tmp_line_y+[tag2id['</s>']]+padding_tag_plus_one)
        tlm_outputs_y_reverse.append(tmp_line_y[::-1]+[tag2id['<s>']]+padding_tag_plus_one)
        tlm_targets_pos.append(tmp_line_pos+[tag2id['</s>']]+padding_tag_plus_one)
        tlm_targets_pos_reverse.append(tmp_line_pos[::-1]+[tag2id['<s>']]+padding_tag_plus_one)
        
        weights.append([1.0] * (l+1-1) + [0.0] * (max_len-l))
        tlm_weights.append([1.0] * (l-1) + [0.0] * (max_len-l+1))
        
    tmp_random = random.randint(0,1)
    if tmp_random:
        s1 = np.random.dirichlet(np.ones(len(tag2id))*10,size=1)-float(1)/len(tag2id) # size here represent batch size!!!
        s2 = -(np.random.dirichlet(np.ones(len(tag2id))*10,size=1)-float(1)/len(tag2id))
        s = (s1+s2)/2
    else:
        s = np.zeros((1,len(tag2id)))
        # weights_reverse and tlm_weights_reverse are the same as weights and tlm_weights
        
 
    # wrong: should run CNN through chars of each word

    for i in range(len(inputs_x_char)):
        for j in range(len(inputs_x_char[i])):
            max_len_char = max([len(x) for x in inputs_x_char[i]])
            new_len_char = [(max_len_char - len(inputs_x_char[i][j])) for j in range(len(inputs_x_char[i]))]
            inputs_x_char[i][j] += [0 for k in range(new_len_char[j])]
        
    return {'enc_inputs': inputs_x,
            'enc_inputs_reverse': inputs_x_reverse,
            'enc_inputs_char': inputs_x_char, # batchsize=1
            'next_enc_inputs': next_inputs_x,
            'next_enc_inputs_reverse': next_inputs_x_reverse,
            'inputs_pos': inputs_pos,
            'inputs_pos_reverse': inputs_pos_reverse,
            'inputs_chunk': inputs_chunk,
            'inputs_chunk_reverse': inputs_chunk_reverse,
            'inputs_case': inputs_case,
            'inputs_case_reverse': inputs_case_reverse,
            'inputs_num': inputs_num,
            'inputs_num_reverse': inputs_num_reverse,
            'targets': outputs_y,
            'targets_reverse': outputs_y_reverse,
            'tlm_targets': tlm_outputs_y,
            'tlm_targets_reverse': tlm_outputs_y_reverse,
            'tlm_targets_pos': tlm_targets_pos,
            'tlm_targets_pos_reverse': tlm_targets_pos_reverse,
            'batch_size': len(inputs_x),
            'weights': weights,
            'tlm_weights': tlm_weights,
            'len': max_len,
            'size': len(x),
            'perturb': s}

def get_batches(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size):
    n = len(x)
    order = range(n)
    z = sorted(zip(order, x, y, pos, chunk, case, num), key=lambda i: len(i[1]))
    order, x, y, pos, chunk, case, num = zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        if (s-t) < batch_size:
            s = t-batch_size
        batches.append(get_batch(x[s:t], y[s:t], pos[s:t], chunk[s:t], case[s:t], num[s:t], word2id, tag2id, pos2id, chunk2id))
        s = t

    return batches, order










def feed_dictionary(model, batch, dropout, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.enc_inputs_char: batch['enc_inputs_char'],
                 model.enc_inputs_reverse: batch['enc_inputs_reverse'],
                 model.next_enc_inputs: batch['next_enc_inputs'],
                 model.next_enc_inputs_reverse: batch['next_enc_inputs_reverse'],
                 model.inputs_pos: batch['inputs_pos'],
                 model.inputs_pos_reverse: batch['inputs_pos_reverse'],
                 model.inputs_chunk: batch['inputs_chunk'],
                 model.inputs_chunk_reverse: batch['inputs_chunk_reverse'],
                 model.inputs_case: batch['inputs_case'],
                 model.inputs_case_reverse: batch['inputs_case_reverse'],
                 model.inputs_num: batch['inputs_num'],
                 model.inputs_num_reverse: batch['inputs_num_reverse'],
                 model.targets: batch['targets'],
                 model.targets_reverse: batch['targets_reverse'],
                 model.tlm_targets: batch['tlm_targets'],
                 model.tlm_targets_reverse: batch['tlm_targets_reverse'],
                 model.tlm_targets_pos: batch['tlm_targets_pos'],
                 model.tlm_targets_pos_reverse: batch['tlm_targets_pos_reverse'],
                 model.weights: batch['weights'],
                 model.tlm_weights: batch['tlm_weights'],
                 model.perturb: batch['perturb']}
    return feed_dict



def create_cell_gru(dim, dropout):
    cell = tf.nn.rnn_cell.GRUCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    return cell



def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def cnn(inp, scope, reuse=False):
    filter_sizes = [2,3] # hyperparameters
    n_filters = 64 # hyperparameters
    dropout = 0.7 # hyperparameters
    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)
    num_words = inp.get_shape().as_list()[0]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                W = tf.get_variable('W', [size, dim, 1, n_filters])
                b = tf.get_variable('b', [n_filters])
                conv = tf.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.reduce_max(h, reduction_indices=1)
                pooled = tf.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.concat(outputs, 1)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [n_filters*len(filter_sizes), 16])
            b = tf.get_variable('b', [16])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1, 16])

    return logits







