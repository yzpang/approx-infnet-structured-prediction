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
from evaluate import *








# InfNet

class InfNet_TLM(object):
    
    def __init__(self, dim_h, tag_size, vocab_size):

        dim_emb = 100
        beta1, beta2 = 0.9, 0.999
        dim_d = 2*dim_h # value of d
        
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.batch_len = tf.placeholder(tf.int32, name='batch_len')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None], name='enc_inputs') # size * len
        self.enc_inputs_reverse = tf.placeholder(tf.int32, [None, None], name='enc_inputs_reverse')
        self.next_enc_inputs = tf.placeholder(tf.int32, [None, None], name='next_enc_inputs') # size * len
        self.next_enc_inputs_reverse = tf.placeholder(tf.int32, [None, None], name='next_enc_inputs_reverse')
        self.weights = tf.placeholder(tf.float32, [None, None], name='weights')
        self.tlm_weights = tf.placeholder(tf.float32, [None, None], name='tlm_weights')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.targets_reverse = tf.placeholder(tf.int32, [None, None], name='targets_reverse')
        self.tlm_targets = tf.placeholder(tf.int32, [None, None], name='tlm_targets')
        self.tlm_targets_reverse = tf.placeholder(tf.int32, [None, None], name='tlm_targets_reverse')
    
        embedding_model = tf.get_variable('embedding', initializer=embedding_global.astype(np.float32))

        def delta(v):
            return tf.norm(v, ord=1)

        inputs = tf.nn.embedding_lookup(embedding_global, self.enc_inputs)
        inputs = tf.cast(inputs, tf.float32)
        
        next_inputs = tf.nn.embedding_lookup(embedding_global, self.next_enc_inputs) 
        next_inputs = tf.cast(next_inputs, tf.float32)
        # but use self.next_enc_inputs as targets in LM
        
        inputs_reverse = tf.nn.embedding_lookup(embedding_global, self.enc_inputs_reverse)
        inputs_reverse = tf.cast(inputs_reverse, tf.float32)
        
        next_inputs_reverse = tf.nn.embedding_lookup(embedding_global, self.next_enc_inputs_reverse) 
        next_inputs_reverse = tf.cast(next_inputs_reverse, tf.float32)
        
        
        
        ''' Implementing TLM 
        - Tag embeddings are L dimensional one-hot vectors // why not just random initialization
        - GRU (paper uses LSTM) language model on the tag sequences
        '''
        
        with tf.variable_scope('tlm_projection'):
            proj_tlm_W = tf.get_variable('tlm_W', [dim_h, tag_size+vocab_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_b = tf.get_variable('tlm_b', [tag_size+vocab_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_W_reverse = tf.get_variable('tlm_W_reverse', [dim_h, tag_size+vocab_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_b_reverse = tf.get_variable('tlm_b_reverse', [tag_size+vocab_size], dtype=tf.float32) # tag_size+vocab_size
        
        
        y_onehot_tlm = tf.one_hot(self.targets, depth=tag_size)
        y_onehot_tlm_reverse = tf.one_hot(self.targets_reverse, depth=tag_size)
        
        with tf.variable_scope('tlm'):
            cell_gru = create_cell_gru(dim_h, self.dropout)
            # initial_state_gru = cell_gru.zero_state(batch_size, dtype=tf.float32)
            outputs_tlm, _ = tf.nn.dynamic_rnn(cell_gru, 
                                               tf.concat([inputs,y_onehot_tlm], axis=-1), # [inputs,y_onehot_tlm]
                                               dtype=tf.float32, scope='tlm')
            outputs_tlm = tf.nn.dropout(outputs_tlm, self.dropout)
            outputs_tlm = tf.reshape(outputs_tlm, [-1, dim_h])

            self.logits_tlm_tmp = tf.matmul(outputs_tlm, proj_tlm_W) + proj_tlm_b
            self.logits_tlm = self.logits_tlm_tmp[:,vocab_size:] # FIX!!!!!!!!!
            self.logits_nextword = self.logits_tlm_tmp[:,:vocab_size]
            
            self.probs_tlm = tf.nn.softmax(self.logits_tlm)
            self.probs_nextword = tf.nn.softmax(self.logits_nextword)
            # self.probs_tlm = self.probs_nextword # delete later


            loss_pretrain_tlm = tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels=tf.reshape(self.tlm_targets, [-1]),
               logits=self.logits_tlm)
            loss_pretrain_tlm *= tf.reshape(self.tlm_weights, [-1])
            loss_pretrain_nextword = tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels=tf.reshape(self.next_enc_inputs, [-1]),
               logits=self.logits_nextword)
            loss_pretrain_nextword *= tf.reshape(self.weights, [-1])

            self.tlm_tot_loss_0 = tf.reduce_sum(loss_pretrain_nextword)
            self.tlm_tot_loss_1 = tf.reduce_sum(loss_pretrain_tlm)
            self.tlm_tot_loss = self.tlm_tot_loss_0 + self.tlm_tot_loss_1  #
            self.tlm_sent_loss_0 = self.tlm_tot_loss_0 / tf.to_float(self.batch_size)
            self.tlm_sent_loss_1 = self.tlm_tot_loss_1 / tf.to_float(self.batch_size)
            self.tlm_sent_loss = self.tlm_tot_loss / tf.to_float(self.batch_size)
        


        with tf.variable_scope('tlm_reverse'):
            cell_gru_reverse = create_cell_gru(dim_h, self.dropout)
            outputs_tlm_reverse, _ = tf.nn.dynamic_rnn(cell_gru_reverse, 
                                               tf.concat([inputs_reverse,y_onehot_tlm_reverse], axis=-1), # [inputs,y_onehot_tlm]
                                               dtype=tf.float32, scope='tlm_reverse')
            outputs_tlm_reverse = tf.nn.dropout(outputs_tlm_reverse, self.dropout)
            outputs_tlm_reverse = tf.reshape(outputs_tlm_reverse, [-1, dim_h])

            
            self.logits_tlm_tmp_reverse = tf.matmul(outputs_tlm_reverse, proj_tlm_W_reverse) + proj_tlm_b_reverse
            self.logits_tlm_reverse = self.logits_tlm_tmp_reverse[:,vocab_size:] # FIX!!!!!!!!!
            self.logits_nextword_reverse = self.logits_tlm_tmp_reverse[:,:vocab_size]
            
            self.probs_tlm_reverse = tf.nn.softmax(self.logits_tlm_reverse)
            self.probs_nextword_reverse = tf.nn.softmax(self.logits_nextword_reverse)
            

            #self.output_0_shape = tf.shape(self.logits_tlm_tmp)
            #self.output_1_shape = tf.shape(self.logits_tlm)

            
            loss_pretrain_tlm_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels=tf.reshape(self.tlm_targets_reverse, [-1]),
               logits=self.logits_tlm_reverse)
            loss_pretrain_tlm_reverse *= tf.reshape(self.tlm_weights, [-1])
            loss_pretrain_nextword_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels=tf.reshape(self.next_enc_inputs_reverse, [-1]),
               logits=self.logits_nextword_reverse)
            loss_pretrain_nextword_reverse *= tf.reshape(self.weights, [-1])
            
            

            
            self.tlm_tot_loss_0_reverse = tf.reduce_sum(loss_pretrain_nextword_reverse)
            self.tlm_tot_loss_1_reverse = tf.reduce_sum(loss_pretrain_tlm_reverse)
            self.tlm_tot_loss_reverse = self.tlm_tot_loss_0_reverse + self.tlm_tot_loss_1_reverse
            self.tlm_sent_loss_0_reverse = self.tlm_tot_loss_0_reverse / tf.to_float(self.batch_size)
            self.tlm_sent_loss_1_reverse = self.tlm_tot_loss_1_reverse / tf.to_float(self.batch_size) 
            self.tlm_sent_loss_reverse = self.tlm_tot_loss_reverse / tf.to_float(self.batch_size)
            
        

        self.tlm_train_loss_0 = self.tlm_sent_loss_0+self.tlm_sent_loss_0_reverse
        self.tlm_train_loss_1 = self.tlm_sent_loss_1+self.tlm_sent_loss_1_reverse

        
        tlm_param = retrive_var(['tlm_projection','tlm','tlm_reverse'])
        self.optimizer_tlm_0 = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.tlm_train_loss_0, var_list=tlm_param)
        self.optimizer_tlm_1 = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.tlm_train_loss_1, var_list=tlm_param)
        
        
        
        ''' Implementing A_phi
        - An RNN that returns a vector at each position of x
        - We can interpret this vector as prob distn over output labels at that position
        - We first try an architecture of BiLSTM for A_phi
        '''
        
        with tf.variable_scope('phi_projection'):
            proj_W = tf.get_variable('W', [2*dim_h, tag_size], dtype=tf.float32) # 2 because of BiLSTM 
            proj_b = tf.get_variable('b', [tag_size], dtype=tf.float32)
        
        with tf.variable_scope('phi'):
            cell_fw = create_cell(dim_h, self.dropout)
            cell_bw = create_cell(dim_h, self.dropout)
            initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32, scope='phi')
            
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.nn.dropout(outputs, self.dropout)
            outputs = tf.reshape(outputs, [-1, 2*dim_h])
            outputs = tf.cast(outputs, tf.float32)

        # affine transformation to get logits
        self.phi_logits = tf.matmul(outputs, proj_W) + proj_b # shape is (batch_size(2)*batch_length, 28)
        self.phi_probs = tf.nn.softmax(self.phi_logits) 
        
        
        





        
        
        
        
        '''
        phi_probs_for_input = tf.reshape(self.phi_probs, [self.batch_size, self.batch_len, tag_size])
        phi_probs_for_input_reverse = tf.reshape(self.phi_probs[::-1,:], [self.batch_size, self.batch_len, tag_size])

        with tf.variable_scope('tlm', reuse=True):
            outputs_tlm_eval, _ = tf.nn.dynamic_rnn(cell_gru, 
                                               tf.concat([inputs,phi_probs_for_input], axis=-1), # [inputs,y_onehot_tlm]
                                               dtype=tf.float32, scope='tlm')
            #outputs_tlm_eval = tf.nn.dropout(outputs_tlm_eval, self.dropout)
            outputs_tlm_eval = tf.reshape(outputs_tlm_eval, [-1, dim_h])

            self.logits_tlm_tmp_eval = tf.matmul(outputs_tlm_eval, proj_tlm_W) + proj_tlm_b
            self.logits_tlm_eval = self.logits_tlm_tmp_eval[:,vocab_size:] # FIX!!!!!!!!!
            self.logits_nextword_eval = self.logits_tlm_tmp_eval[:,:vocab_size]
            
            self.probs_tlm_eval = tf.nn.softmax(self.logits_tlm_eval)
            self.probs_nextword_eval = tf.nn.softmax(self.logits_nextword_eval)
            
        with tf.variable_scope('tlm_reverse', reuse=True):
            outputs_tlm_reverse_eval, _ = tf.nn.dynamic_rnn(cell_gru_reverse, 
                                               tf.concat([inputs_reverse,phi_probs_for_input_reverse], axis=-1), # [inputs,y_onehot_tlm]
                                               dtype=tf.float32, scope='tlm_reverse')
            #outputs_tlm_reverse_eval = tf.nn.dropout(outputs_tlm_reverse_eval, self.dropout)
            outputs_tlm_reverse_eval = tf.reshape(outputs_tlm_reverse_eval, [-1, dim_h])

            
            self.logits_tlm_tmp_reverse_eval = tf.matmul(outputs_tlm_reverse_eval, proj_tlm_W_reverse) + proj_tlm_b_reverse
            self.logits_tlm_reverse_eval = self.logits_tlm_tmp_reverse_eval[:,vocab_size:] # FIX!!!!!!!!!
            self.logits_nextword_reverse_eval = self.logits_tlm_tmp_reverse_eval[:,:vocab_size]
            
            self.probs_tlm_reverse_eval = tf.nn.softmax(self.logits_tlm_reverse_eval)
            self.probs_nextword_reverse_eval = tf.nn.softmax(self.logits_nextword_reverse_eval)
        '''














        ''' Implementing energy function '''
        
        with tf.variable_scope('energy_function'):
            energy_U = tf.get_variable('energy_U', [tag_size, dim_d], dtype=tf.float32)
            energy_W = tf.get_variable('energy_W', [tag_size, tag_size], dtype=tf.float32)
        
        # with tf.variable_scope('energy_feature_proj'):
        #     energy_proj_W = tf.get_variable('energy_proj_W', [2*dim_h, dim_d], dtype=tf.float32) # 2 because of BiLSTM 
        #     energy_proj_b = tf.get_variable('energy_proj_b', [dim_d], dtype=tf.float32)
        
        with tf.variable_scope('energy_feature'):
            cell_fw = create_cell(dim_h, self.dropout)
            cell_bw = create_cell(dim_h, self.dropout)
            initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32, scope='energy_feature')
            
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.nn.dropout(outputs, self.dropout)
            outputs = tf.reshape(outputs, [-1, 2*dim_h])
            outputs = tf.cast(outputs, tf.float32)

        # shape is (batch_size(2)*batch_length, 100)
        energy_feature_vec = outputs # tf.matmul(outputs, energy_proj_W) + energy_proj_b
        

        # SHOULD merge two energy functions into one

        def energy_result(self, x, y, y_unscale_logits, x_nextword_onehot, x_nextword_onehot_reverse):
        
        
            # note that energy_feature_vec will be looped around twice with batch_size 2
            M0 = tf.matmul(energy_U, tf.transpose(energy_feature_vec)) 
            tmp0 = tf.multiply(y, tf.transpose(M0)) # elt-wise
            energy_first_part = tf.reduce_sum(tmp0)
            
            #y_prime = tf.manip.roll(y, shift=1, axis=0)
            #y_prime = tf.concat([[tf.zeros([tag_size])], y_prime[1:]], axis=0) # check y has 28 as last dim
            
            y_prime = tf.concat([[tf.zeros([tag_size])], y[:-1]], axis=0) # check y has 28 as last dim
            tmp1 = tf.matmul(tf.matmul(y_prime, energy_W), tf.transpose(y)) # first y is tricky
            energy_second_part = tf.reduce_sum(tf.diag_part(tmp1))
            old_return = -(energy_first_part+energy_second_part)
                        
           

            '''
            # now implement E_TLM
            tmp_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(y[1:-1],self.probs_tlm_eval[:-2]), axis=-1))
            tmp_energy_tlm = tf.reduce_sum(tmp_energy_tlm)
            old_return += 0.075 * tmp_energy_tlm

        
#             loss_lm = tf.nn.softmax_cross_entropy_with_logits_v2(
#             labels=self.probs_nextword[:-1],
#             logits=self.logits_nextword_eval[:-1])
#             loss_lm = 0.003 * tf.reduce_sum(tf.reshape(loss_lm,[-1]))
            
#             old_return += loss_lm

            # # now implement E_nextword
            # tmp_energy_nextword = -tf.log(tf.reduce_sum(tf.multiply(x_nextword_onehot[:-1],self.probs_nextword[:-1]), axis=-1))
            # tmp_energy_nextword = tf.reduce_sum(tmp_energy_nextword)
            # old_return += 1000.0 * tmp_energy_nextword


            # # backward LM
            # # the line below only works for batch_size == 1
            tmp_energy_tlm_reverse = -tf.log(tf.reduce_sum(tf.multiply(y[::-1][1:-1],self.probs_tlm_reverse_eval[:-2]), axis=-1))
            tmp_energy_tlm_reverse = tf.reduce_sum(tmp_energy_tlm_reverse)
            old_return += 0.075 * tmp_energy_tlm_reverse

            # tmp_energy_nextword_reverse = -tf.log(tf.reduce_sum(tf.multiply(x_nextword_onehot_reverse[:-1],self.probs_nextword_reverse[:-1]), axis=-1))
            # tmp_energy_nextword_reverse = tf.reduce_sum(tmp_energy_nextword_reverse)
            # old_return += 0.07 * tmp_energy_nextword_reverse
            '''
            
            
            
            return old_return #+ loss_lm + loss_lm_reverse


        def energy_result_gold(self, x, y, y_unscale_logits, x_nextword_onehot, x_nextword_onehot_reverse):
        
        
            # note that energy_feature_vec will be looped around twice with batch_size 2
            M0 = tf.matmul(energy_U, tf.transpose(energy_feature_vec)) 
            tmp0 = tf.multiply(y, tf.transpose(M0)) # elt-wise
            energy_first_part = tf.reduce_sum(tmp0)
            
            #y_prime = tf.manip.roll(y, shift=1, axis=0)
            #y_prime = tf.concat([[tf.zeros([tag_size])], y_prime[1:]], axis=0) # check y has 28 as last dim
            
            y_prime = tf.concat([[tf.zeros([tag_size])], y[:-1]], axis=0) # check y has 28 as last dim
            tmp1 = tf.matmul(tf.matmul(y_prime, energy_W), tf.transpose(y)) # first y is tricky
            energy_second_part = tf.reduce_sum(tf.diag_part(tmp1))
            old_return = -(energy_first_part+energy_second_part)
                        
           

            '''
            # now implement E_TLM
            tmp_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(y[1:-1],self.probs_tlm[:-2]), axis=-1))
            tmp_energy_tlm = tf.reduce_sum(tmp_energy_tlm)
            old_return += 0.075 * tmp_energy_tlm
            
            tmp_energy_tlm_reverse = -tf.log(tf.reduce_sum(tf.multiply(y[::-1][1:-1],self.probs_tlm_reverse[:-2]), axis=-1))
            tmp_energy_tlm_reverse = tf.reduce_sum(tmp_energy_tlm_reverse)
            old_return += 0.075 * tmp_energy_tlm_reverse
            '''
        
            return old_return 



        ''' Implementing phi '''
        
        y_onehot = tf.one_hot(self.targets, depth=tag_size)
        y_onehot = tf.reshape(y_onehot, [-1, tag_size])
        tmp_delta_0 = tf.reduce_sum(self.phi_probs - y_onehot, axis=-1)
        tmp_delta_0 *= tf.reshape(self.weights,[-1])
        
        x_nextword_onehot = tf.one_hot(self.next_enc_inputs, depth=vocab_size)
        x_nextword_onehot = tf.reshape(x_nextword_onehot, [-1, vocab_size])
        
        x_nextword_onehot_reverse = tf.one_hot(self.next_enc_inputs_reverse, depth=vocab_size)
        x_nextword_onehot_reverse = tf.reshape(x_nextword_onehot_reverse, [-1, vocab_size])


        extra_reg_term = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]),logits=self.phi_logits)
        extra_reg_term *= tf.reshape(self.weights, [-1])
        extra_reg_term = tf.reduce_sum(extra_reg_term) / tf.to_float(self.batch_size)


    
    
    

        
        
        
        

        # self.loss_phi *= tf.reshape(self.weights, [-1])
        # something like this 
        loss_phi = delta(tmp_delta_0) - energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse) \
            + energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse)
        loss_phi = -loss_phi
        # loss_phi = tf.minimum(900.0, loss_phi)
        self.loss_phi = tf.maximum(loss_phi, 0.0) + 0.3 * extra_reg_term #+ loss_lm + loss_lm_reverse # + 

        loss_theta = delta(tmp_delta_0) - energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse) \
            + energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse) #+ 0.0001 * retrive_var_regularize(['energy_function','energy_feature_proj','energy_feature']) # regularization
        self.loss_theta = tf.maximum(loss_theta, 0.0)
        
        
        
        
        
        ''' Optimization '''
        
        phi = retrive_var(['phi_projection','phi'])
        theta = retrive_var(['energy_function','energy_feature']) 
        self.optimizer_phi = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_phi, var_list=phi)
        self.optimizer_theta = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_theta, var_list=theta)
 
        
        
        
        ''' Implementing psi '''
        
        self.loss_psi = energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse)
        psi = retrive_var(['phi_projection','phi'])
        self.optimizer_psi = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_psi, var_list=psi)
        
        
        
        self.saver = tf.train.Saver()
        
        
        



def create_model_infnet_tlm(sess, dim_h, n_tag, vocab_size, load_model=False, model_path=''):
    model = InfNet_TLM(dim_h, n_tag, vocab_size)
    if load_model:
        print('Loading model from ...')
        #tf.reset_default_graph()
        #sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, model_path)
        # sess.run(tf.initialize_variables(tf.report_uninitialized_variables(tf.all_variables)))
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    
    return model


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    args = load_arguments()

    print('loaded arguments')


    train_data = load_sent(args.train)
    dev_data = load_sent(args.dev)
    test_data = load_sent(args.test)



    embedding2id, id2embedding, embedding_old = load_embedding(args.embedding) # change embedding_global



    tag2id, id2tag = convert_tag_to_id(train_data)

    train_data = preprocess_data_according_to_rules(train_data, embedding_old)
    dev_data = preprocess_data_according_to_rules(dev_data, embedding_old)
    test_data = preprocess_data_according_to_rules(test_data, embedding_old)

    word2id, id2word = construct_word_id(train_data+dev_data+test_data) 

    embedding_global = construct_embedding(len(word2id), 100, embedding_old, word2id)

    x_train, y_train = turn_data_into_x_y(train_data, word2id)
    x_dev, y_dev = turn_data_into_x_y(dev_data, word2id)
    x_test, y_test = turn_data_into_x_y(test_data, word2id)


    x_train_full = x_train
    y_train_full = y_train

    # STEP x - TRAINING PHI AND THETA

    steps_per_checkpoint = args.steps_per_checkpoint
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    batch_size = 1 # for now, only support batch size 1 to avoid mistakes
    dropout = args.dropout
    load_model = str2bool(args.load_model)
    model_name = args.model
    dim_h = args.dim_h

    with tf.Graph().as_default():
        with tf.Session() as sess:

            # model = create_model_infnet(sess, 100, len(tag2id)) # use 100
            model = create_model_infnet_tlm(sess, dim_h, len(tag2id), len(word2id), load_model, model_name) # create right model!

            if True: # training
                batches, _ = get_batches(x_train, y_train, word2id, tag2id, batch_size)
                random.shuffle(batches)

                start_time = time.time()
                step = 0
                loss_phi, loss_theta = 0.0, 0.0
                #best_dev = float('-inf')
                best_dev = 0.00

                for epoch in range(max_epochs):
                    print('----------------------------------------------------')
                    print('epoch %d, learning_rate %f' % (epoch + 1, learning_rate))

                    for batch in batches:
                        
                        if batch['size'] == batch_size:
                        
                        
                            feed_dict_tmp = feed_dictionary(model, batch, dropout, learning_rate)
                            
    #                         tmp0, tmp1 = sess.run([model.probs_tlm, model.probs_tlm_eval],
    #                            feed_dict=feed_dict_tmp)
    #                         print(tmp0-tmp1)
                                      
                            
                            step_loss_phi, _ = sess.run([model.loss_phi, model.optimizer_phi],
                                feed_dict=feed_dict_tmp)                   
                            step_loss_theta, _ = sess.run([model.loss_theta, model.optimizer_theta],
                                feed_dict=feed_dict_tmp)

                            step += 1
                            loss_phi += step_loss_phi / steps_per_checkpoint
                            loss_theta += step_loss_theta / steps_per_checkpoint

                            if step % steps_per_checkpoint == 0:
                                print('step %d, time %.0fs, loss_phi %.2f, loss_theta %.2f' \
                                    % (step, time.time() - start_time, loss_phi, loss_theta))
                                loss_phi, loss_theta = 0.0, 0.0
                                
                    
    #                print('------ ... -> saving model...')
    #                model.saver.save(sess, model_name)
                    
                                #acc, _ = evaluate(sess, model, x_dev, y_dev, word2id, tag2id, batch_size)
                                #print('-- dev acc: %.2f' % acc)
                
                            if step % (1*steps_per_checkpoint) == 0: # MODIFY LATER
                                acc, _ = evaluate(sess, model, x_dev, y_dev, word2id, tag2id, batch_size)
                                print('-- dev acc: %.2f' % acc)
                                
                                if acc > best_dev:
                                    best_dev = acc
                                    print('------ best dev acc so far -> saving model...')
                                    model.saver.save(sess, model_name)
                                    
                                    acc, _ = evaluate(sess, model, x_test, y_test, word2id, tag2id, batch_size)
                                    print('-- test acc: %.2f' % acc)
                                    
                                
                    acc, _ = evaluate(sess, model, x_train, y_train, word2id, tag2id, batch_size)
                    print('-- train acc: %.2f' % acc)

                    acc, _ = evaluate(sess, model, x_dev, y_dev, word2id, tag2id, batch_size)
                    print('-- dev acc: %.2f' % acc)
                    
                    acctest, _ = evaluate(sess, model, x_test, y_test, word2id, tag2id, batch_size)
                    print('-- test acc: %.2f' % acctest)

                    if acc > best_dev:
                        best_dev = acc
                        print('------ best dev acc so far -> saving model...')
                        model.saver.save(sess, model_name)
                    
    #                 # acc, _ = evaluate(sess, model, test_data, window)
    #                 # print('-- test acc: %.2f' % acc)

                    if int(time.time()-start_time) > 14000:
                        print('=== saving model after 14000 seconds -> saving')
                        model.saver.save(sess, model_name+'-contd')



