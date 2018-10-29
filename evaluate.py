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
from utils import *






def evaluate_tlm(sess, model, x, y, word2id, tag2id, batch_size):
    batches, _ = get_batches(x, y, word2id, tag2id, batch_size)
    tot_loss_0, tot_loss_1, tot_loss_0_reverse, tot_loss_1_reverse, n_words = 0, 0, 0, 0, 0

    for batch in batches:
        if batch['size'] == batch_size:
            tmp_tot_loss_0, tmp_tot_loss_1, tmp_tot_loss_0_reverse, tmp_tot_loss_1_reverse = sess.run([model.tlm_tot_loss_0, model.tlm_tot_loss_1, model.tlm_tot_loss_0_reverse, model.tlm_tot_loss_1_reverse],
                feed_dict={model.batch_size: batch['size'],
                           model.enc_inputs: batch['enc_inputs'],
                           model.enc_inputs_reverse: batch['enc_inputs_reverse'],
                           model.next_enc_inputs: batch['next_enc_inputs'],
                           model.next_enc_inputs_reverse: batch['next_enc_inputs_reverse'],
                           model.batch_len: batch['len'],
                           model.targets: batch['targets'],
                           model.targets_reverse: batch['targets_reverse'],
                           model.tlm_targets: batch['tlm_targets'],
                           model.tlm_targets_reverse: batch['tlm_targets_reverse'],
                           model.weights: batch['weights'],
                           model.tlm_weights: batch['tlm_weights'],
                           model.dropout: 1.0})
            tot_loss_0 += tmp_tot_loss_0
            tot_loss_1 += tmp_tot_loss_1
            tot_loss_0_reverse += tmp_tot_loss_0_reverse
            tot_loss_1_reverse += tmp_tot_loss_1_reverse
            
            n_words += np.sum(batch['weights'])

    return np.exp(tot_loss_0 / n_words), np.exp(tot_loss_1 / n_words), np.exp(tot_loss_0_reverse / n_words), np.exp(tot_loss_1_reverse / n_words), 


def evaluate(sess, model, x, y, word2id, tag2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, word2id, tag2id, batch_size)
    y = []
    
    same = 0
    ttl = 0
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
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















def evaluate_print(sess, model, x, y, word2id, tag2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, word2id, tag2id, batch_size)
    y = []
    
    same = 0
    ttl = 0
    
    acc_y = []
    acc_y_hat = []
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
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



