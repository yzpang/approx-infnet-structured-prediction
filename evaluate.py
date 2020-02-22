import random
import re
import sys
import time
import os
import os.path
import string
import subprocess

import numpy as np
from scipy import stats, spatial
from sklearn.decomposition import TruncatedSVD
import tensorflow.compat.v1 as tf

from options import load_arguments
from utils import *





def evaluate_tlm(sess, model, x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size):
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size)
    tot_loss_0, tot_loss_1, tot_loss_2, tot_loss_0_reverse, tot_loss_1_reverse, tot_loss_2_reverse, n_words = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    for batch in batches:
        if batch['size'] == batch_size:
            tmp_tot_loss_1, tmp_tot_loss_2, tmp_tot_loss_1_reverse, tmp_tot_loss_2_reverse = sess.run([model.tlm_tot_loss_1, model.tlm_tot_loss_2,
                model.tlm_tot_loss_1_reverse, model.tlm_tot_loss_2_reverse],
                feed_dict={model.batch_size: batch['size'],
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
                           model.batch_len: batch['len'],
                           model.targets: batch['targets'],
                           model.targets_reverse: batch['targets_reverse'],
                           model.tlm_targets: batch['tlm_targets'],
                           model.tlm_targets_reverse: batch['tlm_targets_reverse'],
                           model.tlm_targets_pos: batch['tlm_targets_pos'],
                           model.tlm_targets_pos_reverse: batch['tlm_targets_pos_reverse'],
                           model.weights: batch['weights'],
                           model.tlm_weights: batch['tlm_weights'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1})
            #tot_loss_0 += tmp_tot_loss_0
            tot_loss_1 += tmp_tot_loss_1
            tot_loss_2 += tmp_tot_loss_2
            #tot_loss_0_reverse += tmp_tot_loss_0_reverse
            tot_loss_1_reverse += tmp_tot_loss_1_reverse
            tot_loss_2_reverse += tmp_tot_loss_2_reverse
            
            n_words += np.sum(batch['weights'])

    return np.exp(tot_loss_1 / n_words), np.exp(tot_loss_2 / n_words),  np.exp(tot_loss_1_reverse / n_words), np.exp(tot_loss_2_reverse / n_words)#, np.exp(tot_loss_2_reverse / n_words)



def evaluate(sess, model, x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size)
    
    y = []
    
    same = 0
    ttl = 0
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.enc_inputs_char: batch['enc_inputs_char'],
                           model.inputs_pos: batch['inputs_pos'],
                           model.inputs_chunk: batch['inputs_chunk'],
                           model.inputs_case: batch['inputs_case'],
                           model.inputs_num: batch['inputs_num'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1.0})

            # shape is (batch_size(4), batch_length, 28)
            probs = probs.reshape((batch['size'],batch['len'],len(tag2id)))

            # shape is (batch_size(4)*batch_length)
            wt = np.array(batch['weights'])
            wt = wt.reshape(batch['size']*batch['len'])

            y = np.array(batch['targets'])
            y = y.reshape(batch['size']*batch['len'])

            y_hat = [np.argmax(p) for i in range(batch['size']) for p in probs[i]]

            for i in range(len(wt)):
                if wt[i] and (y[i] != tag2id['<s>']) and (y[i] != tag2id['</s>']):
                    if y[i] == y_hat[i]:
                        same += 1
                    ttl += 1
        
#         y.append(batch['targets'][0])
#         probs += p.tolist()
#     y_hat = 
#     same = [p == q for p, q in zip(y, y_hat)]

    return 100.0 * (same) / ttl, probs















def evaluate_print(sess, model, x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size)
    y = []
    
    same = 0
    ttl = 0
    
    acc_y = []
    acc_y_hat = []
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.inputs_pos: batch['inputs_pos'],
                           model.inputs_chunk: batch['inputs_chunk'],
                           model.inputs_case: batch['inputs_case'],
                           model.inputs_num: batch['inputs_num'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1.0})

            # shape is (batch_size(4), batch_length, 28)
            probs = probs.reshape((batch['size'],batch['len'],len(tag2id)))

            # shape is (batch_size(4)*batch_length)
            wt = np.array(batch['weights'])
            wt = wt.reshape(batch['size']*batch['len'])

            y = np.array(batch['targets'])
            y = y.reshape(batch['size']*batch['len'])

            y_hat = [np.argmax(p) for i in range(batch['size']) for p in probs[i]]

            for i in range(len(wt)):
                if wt[i]:
                    if y[i] == y_hat[i]:
                        same += 1
                    ttl += 1
                    
            acc_y.append(y)
            acc_y_hat.append(y_hat)
        
#         y.append(batch['targets'][0])
#         probs += p.tolist()
#     y_hat = 
#     same = [p == q for p, q in zip(y, y_hat)]

    return 100.0 * (same) / ttl, probs, batches, acc_y, acc_y_hat













def evaluate_print(sess, model, x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, char2id, tag2id, pos2id, chunk2id, batch_size)
    y = []
    
    same = 0
    ttl = 0
    
    acc_y = []
    acc_y_hat = []
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.enc_inputs_char: batch['enc_inputs_char'],
                           model.inputs_pos: batch['inputs_pos'],
                           model.inputs_chunk: batch['inputs_chunk'],
                           model.inputs_case: batch['inputs_case'],
                           model.inputs_num: batch['inputs_num'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1.0})

            # shape is (batch_size(4), batch_length, 28)
            probs = probs.reshape((batch['size'],batch['len'],len(tag2id)))

            # shape is (batch_size(4)*batch_length)
            wt = np.array(batch['weights'])
            wt = wt.reshape(batch['size']*batch['len'])

            y = np.array(batch['targets'])
            y = y.reshape(batch['size']*batch['len'])

            y_hat = [np.argmax(p) for i in range(batch['size']) for p in probs[i]]

            for i in range(len(wt)):
                if wt[i] and (y[i] != tag2id['<s>']) and (y[i] != tag2id['</s>']):
                    if y[i] == y_hat[i]:
                        same += 1
                    ttl += 1
                    
            acc_y.append(y)
            acc_y_hat.append(y_hat)
        
#         y.append(batch['targets'][0])
#         probs += p.tolist()
#     y_hat = 
#     same = [p == q for p, q in zip(y, y_hat)]

    return 100.0 * (same) / ttl, probs, batches, acc_y, acc_y_hat






def compute_f1(probs_test, batches_test, acc_y_test, acc_y_hat_test, id2x):
    id2word = id2x[0]
    id2pos = id2x[1]
    id2tag = id2x[2]
    store_lst = []
    for bn in range(len(batches_test)):
        batch = batches_test[bn]
        for i in range(batch['len']):
            store_word_id = batch['enc_inputs'][0][i]
            if store_word_id not in [1,2]:
                store_word = id2word[store_word_id]
                store_pos = id2pos[batch['inputs_pos'][0][i]]
                store_real_tag = id2tag[acc_y_test[bn][i]]
                store_predicted_tag = id2tag[acc_y_hat_test[bn][i]]
                store_lst.append([store_word,store_pos,store_real_tag,store_predicted_tag])
        store_lst.append([])

    write_file_name = 'ner_eval_outputs.txt'
    with open(write_file_name, 'w') as f:
        for x in store_lst:
            if len(x) == 0:
                f.write('\n')
            else:
                assert len(x) == 4
                write_str = x[0] + ' ' + x[1] + ' ' + x[2] + ' ' + x[3]
                f.write(write_str+'\n')
                
    bash_command = 'perl conlleval < ' + write_file_name + ' > bash_result.out'
    output = subprocess.check_output(['bash','-c', bash_command])
    with open('bash_result.out') as f:
        tmp = f.readlines()

    return float(tmp[1][-6:-1])



