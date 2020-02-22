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
from utils import *
from evaluate import *






# InfNet

class InfNet_TLM(object):
    
    # dim_h 100, tag_size 28
    def __init__(self, dim_h, tag_size, pos_size, chunk_size, vocab_size):

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
        self.inputs_pos = tf.placeholder(tf.int32, [None, None], name='inputs_pos')
        self.inputs_pos_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_pos_reverse')
        self.inputs_chunk = tf.placeholder(tf.int32, [None, None], name='inputs_chunk')
        self.inputs_chunk_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_chunk_reverse')
        self.inputs_case = tf.placeholder(tf.int32, [None, None], name='inputs_case')
        self.inputs_case_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_case_reverse')
        self.inputs_num = tf.placeholder(tf.int32, [None, None], name='inputs_num')
        self.inputs_num_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_num_reverse')
        
        self.enc_inputs_char = tf.placeholder(tf.int32, [None, None, None], name='enc_inputs_char') # size * len
        
        self.weights = tf.placeholder(tf.float32, [None, None], name='weights')
        self.tlm_weights = tf.placeholder(tf.float32, [None, None], name='tlm_weights')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.targets_reverse = tf.placeholder(tf.int32, [None, None], name='targets_reverse')
        self.tlm_targets = tf.placeholder(tf.int32, [None, None], name='tlm_targets')
        self.tlm_targets_reverse = tf.placeholder(tf.int32, [None, None], name='tlm_targets_reverse')
        self.tlm_targets_pos = tf.placeholder(tf.int32, [None, None], name='tlm_targets_pos')
        self.tlm_targets_pos_reverse = tf.placeholder(tf.int32, [None, None], name='tlm_targets_pos_reverse')
        
        self.perturb = tf.placeholder(tf.float32, [None, None], name='perturb')
    
        embedding_model = tf.get_variable('embedding', initializer=embedding_global.astype(np.float32))
        
        embedding_model_char = tf.get_variable('embedding_char', initializer=embedding_char_global.astype(np.float32))

        def delta(v):
            return tf.norm(v, ord=1)

        inputs = tf.nn.embedding_lookup(embedding_model, self.enc_inputs)
        inputs = tf.cast(inputs, tf.float32)
        
        next_inputs = tf.nn.embedding_lookup(embedding_model, self.next_enc_inputs) 
        next_inputs = tf.cast(next_inputs, tf.float32)
        # but use self.next_enc_inputs as targets in LM
        
        inputs_reverse = tf.nn.embedding_lookup(embedding_model, self.enc_inputs_reverse)
        inputs_reverse = tf.cast(inputs_reverse, tf.float32)
        
        next_inputs_reverse = tf.nn.embedding_lookup(embedding_model, self.next_enc_inputs_reverse) 
        next_inputs_reverse = tf.cast(next_inputs_reverse, tf.float32)
        
        
    
        inputs_char = tf.nn.embedding_lookup(embedding_model_char, self.enc_inputs_char)
        inputs_char = tf.cast(inputs_char, tf.float32)
        
        
        ''' Implementing TLM 
        - Tag embeddings are L dimensional one-hot vectors // why not just random initialization
        - GRU (paper uses LSTM) language model on the tag sequences
        '''
        
        with tf.variable_scope('tlm_projection'):
            proj_tlm_W = tf.get_variable('tlm_W', [dim_h, pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_b = tf.get_variable('tlm_b', [pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_W_reverse = tf.get_variable('tlm_W_reverse', [dim_h, pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_b_reverse = tf.get_variable('tlm_b_reverse', [pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
        
        
        
        

        
        
        y_onehot_tlm = tf.one_hot(self.targets, depth=tag_size) + self.perturb
        y_onehot_tlm_reverse = tf.one_hot(self.targets_reverse, depth=tag_size) + self.perturb
        
        
        inputs_pos_onehot = tf.one_hot(self.inputs_pos, depth=pos_size)
        inputs_pos_onehot_reverse = tf.one_hot(self.inputs_pos_reverse, depth=pos_size)
        inputs_chunk_onehot = tf.one_hot(self.inputs_chunk, depth=chunk_size)
        inputs_chunk_onehot_reverse = tf.one_hot(self.inputs_chunk_reverse, depth=chunk_size)
        inputs_case_onehot = tf.one_hot(self.inputs_case, depth=2) # changed from 4 to 2
        inputs_case_onehot_reverse = tf.one_hot(self.inputs_case_reverse, depth=2)
        inputs_num_onehot = tf.one_hot(self.inputs_num, depth=2)
        inputs_num_onehot_reverse = tf.one_hot(self.inputs_num_reverse, depth=2)
        

        # self.output_0_shape = tf.shape(inputs)
        # self.output_1_shape = tf.shape(y_onehot_tlm)
        # self.output_2_shape = tf.shape(inputs_pos_onehot)
        # self.output_3_shape = tf.shape(inputs_chunk_onehot)
        
        
        
        
#         with tf.variable_scope('tlm'):
#             cell_gru = create_cell(dim_h, self.dropout) # lstm actually
#             # initial_state_gru = cell_gru.zero_state(batch_size, dtype=tf.float32)
#             outputs_tlm, _ = tf.nn.dynamic_rnn(cell_gru, 
#                                                tf.concat([inputs,y_onehot_tlm,inputs_pos_onehot], axis=-1), # [inputs,y_onehot_tlm]
#                                                dtype=tf.float32, scope='tlm')
#             outputs_tlm = tf.nn.dropout(outputs_tlm, self.dropout)
#             outputs_tlm = tf.reshape(outputs_tlm, [-1, dim_h])

#             self.logits_tlm_tmp = tf.matmul(outputs_tlm, proj_tlm_W) + proj_tlm_b
#             self.logits_tlm = self.logits_tlm_tmp[:,pos_size:] # FIX!!!!!!!!!
#             # self.logits_nextword = self.logits_tlm_tmp[:,:vocab_size]
#             self.logits_pos = self.logits_tlm_tmp[:,:pos_size]

#             self.probs_tlm = tf.nn.softmax(self.logits_tlm)
#             # self.probs_nextword = tf.nn.softmax(self.logits_nextword)
#             self.probs_pos = tf.nn.softmax(self.logits_pos)

        
#             loss_pretrain_tlm = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets, [-1]),
#                logits=self.logits_tlm)
#             loss_pretrain_tlm *= tf.reshape(self.tlm_weights, [-1])
# #             loss_pretrain_nextword = tf.nn.sparse_softmax_cross_entropy_with_logits(
# #                labels=tf.reshape(self.next_enc_inputs, [-1]),
# #                logits=self.logits_nextword)
# #             loss_pretrain_nextword *= tf.reshape(self.weights, [-1])
#             loss_pretrain_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets_pos, [-1]),
#                logits=self.logits_pos)
#             loss_pretrain_pos *= tf.reshape(self.tlm_weights, [-1])



#         with tf.variable_scope('tlm_reverse'):
#             cell_gru_reverse = create_cell(dim_h, self.dropout)
#             outputs_tlm_reverse, _ = tf.nn.dynamic_rnn(cell_gru_reverse, 
#                                                tf.concat([inputs_reverse,y_onehot_tlm_reverse,inputs_pos_onehot_reverse], axis=-1), # [inputs,y_onehot_tlm]
#                                                dtype=tf.float32, scope='tlm_reverse')
#             outputs_tlm_reverse = tf.nn.dropout(outputs_tlm_reverse, self.dropout)
#             outputs_tlm_reverse = tf.reshape(outputs_tlm_reverse, [-1, dim_h])
                    

            
#             self.logits_tlm_tmp_reverse = tf.matmul(outputs_tlm_reverse, proj_tlm_W_reverse) + proj_tlm_b_reverse
#             self.logits_tlm_reverse = self.logits_tlm_tmp_reverse[:,pos_size:] # FIX!!!!!!!!!
#             # self.logits_nextword_reverse = self.logits_tlm_tmp_reverse[:,:vocab_size]
#             self.logits_pos_reverse = self.logits_tlm_tmp_reverse[:,:pos_size]
            
#             self.probs_tlm_reverse = tf.nn.softmax(self.logits_tlm_reverse)
#             # self.probs_nextword_reverse = tf.nn.softmax(self.logits_nextword_reverse)
#             self.probs_pos_reverse = tf.nn.softmax(self.logits_pos_reverse)
            



#             loss_pretrain_tlm_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets_reverse, [-1]),
#                logits=self.logits_tlm_reverse)
#             loss_pretrain_tlm_reverse *= tf.reshape(self.tlm_weights, [-1])
# #             loss_pretrain_nextword_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
# #                labels=tf.reshape(self.next_enc_inputs_reverse, [-1]),
# #                logits=self.logits_nextword_reverse)
# #             loss_pretrain_nextword_reverse *= tf.reshape(self.weights, [-1])
#             loss_pretrain_pos_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets_pos_reverse, [-1]),
#                logits=self.logits_pos_reverse)
#             loss_pretrain_pos_reverse *= tf.reshape(self.tlm_weights, [-1])
            

            
#         #     #self.tlm_tot_loss_0 = tf.reduce_sum(loss_pretrain_nextword)
#         #     self.tlm_tot_loss_1 = tf.reduce_sum(loss_pretrain_tlm)
#         #     self.tlm_tot_loss_2 = tf.reduce_sum(loss_pretrain_pos)
#         #     self.tlm_tot_loss = self.tlm_tot_loss_1 + self.tlm_tot_loss_2 #+ self.tlm_tot_loss_2 #
#         #     self.tlm_sent_loss_1 = self.tlm_tot_loss_1 / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss_2 = self.tlm_tot_loss_2 / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss = self.tlm_tot_loss / tf.to_float(self.batch_size)
            
#         #     #self.tlm_tot_loss_0_reverse = tf.reduce_sum(loss_pretrain_nextword_reverse)
#         #     self.tlm_tot_loss_1_reverse = tf.reduce_sum(loss_pretrain_tlm_reverse)
#         #     self.tlm_tot_loss_2_reverse = tf.reduce_sum(loss_pretrain_pos_reverse)
#         #     self.tlm_tot_loss_reverse = self.tlm_tot_loss_1_reverse + self.tlm_tot_loss_2_reverse #+ self.tlm_tot_loss_2_reverse
#         #     self.tlm_sent_loss_1_reverse = self.tlm_tot_loss_1_reverse / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss_2_reverse = self.tlm_tot_loss_2_reverse / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss_reverse = self.tlm_tot_loss_reverse / tf.to_float(self.batch_size)
            
            
#         # self.tlm_train_loss_1 = self.tlm_sent_loss_1+self.tlm_sent_loss_1_reverse
#         # self.tlm_train_loss_2 = self.tlm_sent_loss_2+self.tlm_sent_loss_2_reverse

        
#         # tlm_param = retrive_var(['tlm_projection','tlm','tlm_reverse'])
#         # self.optimizer_tlm_1 = tf.train.AdamOptimizer(self.learning_rate,
#         #     beta1, beta2).minimize(self.tlm_train_loss_1, var_list=tlm_param)
#         # self.optimizer_tlm_2 = tf.train.AdamOptimizer(self.learning_rate,
#         #     beta1, beta2).minimize(self.tlm_train_loss_2, var_list=tlm_param)



        
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
            
            
            logits_cnn = cnn(inputs_char[0],'phi') # batch size 1
            logits_cnn = tf.cast(logits_cnn, tf.float32)
            logits_cnn = tf.expand_dims(logits_cnn, 0)
            self.shape0 = tf.shape(logits_cnn) # [20,64]
            
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
                tf.concat([inputs, logits_cnn, inputs_pos_onehot,inputs_chunk_onehot,inputs_case_onehot,inputs_num_onehot], axis=-1),  #inputs_pos_onehot
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32, scope='phi')
            

            
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.nn.dropout(outputs, self.dropout)
            outputs = tf.reshape(outputs, [-1, 2*dim_h])
            outputs = tf.cast(outputs, tf.float32)
            
            self.shape1 = tf.shape(outputs) # [20,256]
            

        # affine transformation to get logits
        self.phi_logits = tf.matmul(tf.concat([outputs], axis=-1), proj_W) + proj_b # shape is (batch_size(2)*batch_length, 28)
        self.phi_probs = tf.nn.softmax(self.phi_logits) # changed from sigmoid to softmax
        # But the thing is some of the logits do not count - we need to deal with it
        
        
        
        
        
        
        
        
        
        
        
        
        





        phi_probs_for_input = tf.reshape(self.phi_probs, [self.batch_size, self.batch_len, tag_size])
        phi_probs_for_input_reverse = tf.reshape(self.phi_probs[::-1,:], [self.batch_size, self.batch_len, tag_size])







        ''' Implementing energy function '''
        
        with tf.variable_scope('energy_function'):
            energy_U = tf.get_variable('energy_U', [tag_size, dim_d+50], dtype=tf.float32)
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








        
        with tf.variable_scope('energy_feature_pos'):
            cell_fw_pos = create_cell(25, self.dropout)
            cell_bw_pos = create_cell(25, self.dropout)
            initial_state_fw_pos = cell_fw_pos.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw_pos = cell_bw_pos.zero_state(batch_size, dtype=tf.float32)
            outputs_pos, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_pos, cell_bw_pos, inputs_pos_onehot, 
                initial_state_fw=initial_state_fw_pos,
                initial_state_bw=initial_state_bw_pos,
                dtype=tf.float32, scope='energy_feature_pos')
            
            outputs_pos = tf.concat(outputs_pos, axis=-1)
            outputs_pos = tf.nn.dropout(outputs_pos, self.dropout)
            outputs_pos = tf.reshape(outputs_pos, [-1, 2*25])
            outputs_pos = tf.cast(outputs_pos, tf.float32)




        # shape is (batch_size(2)*batch_length, 100)
        energy_feature_vec = tf.concat([outputs,outputs_pos],axis=-1) #tf.matmul(outputs, energy_proj_W) + energy_proj_b

        # concat with pos feature vec
        # fix energy_U etc dimension 

        
        def energy_result(self, x, y, y_unscale_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse):

        
            # note that energy_feature_vec will be looped around twice with batch_size 2
            M0 = tf.matmul(energy_U, tf.transpose(energy_feature_vec)) 
            tmp0 = tf.multiply(y, tf.transpose(M0)) # elt-wise
            energy_first_part = tf.reduce_sum(tmp0)
            
            #y_prime = tf.manip.roll(y, shift=1, axis=0)
            #y_prime = tf.concat([[tf.zeros([tag_size])], y_prime[1:]], axis=0) # check y has 28 as last dim
            
            y_prime = y[:-1] # check y has 28 as last dim
            tmp1 = tf.multiply(tf.matmul(y_prime, energy_W), y[1:]) # first y is tricky
            energy_second_part = tf.reduce_sum(tmp1)
            old_return = -(energy_first_part+energy_second_part)


            return old_return



        def energy_result_gold(self, x, y, y_unscale_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse):

        
            # note that energy_feature_vec will be looped around twice with batch_size 2
            M0 = tf.matmul(energy_U, tf.transpose(energy_feature_vec)) 
            tmp0 = tf.multiply(y, tf.transpose(M0)) # elt-wise
            energy_first_part = tf.reduce_sum(tmp0)
            
            #y_prime = tf.manip.roll(y, shift=1, axis=0)
            #y_prime = tf.concat([[tf.zeros([tag_size])], y_prime[1:]], axis=0) # check y has 28 as last dim
            
            y_prime = y[:-1] # check y has 28 as last dim
            tmp1 = tf.multiply(tf.matmul(y_prime, energy_W), y[1:]) # first y is tricky
            energy_second_part = tf.reduce_sum(tmp1)
            old_return = -(energy_first_part+energy_second_part)
            

            return old_return

        
        
        ''' Implementing phi and theta '''
        
        y_onehot = tf.one_hot(self.targets, depth=tag_size)
        y_onehot = tf.reshape(y_onehot, [-1, tag_size])
        tmp_delta_0 = tf.reduce_sum(self.phi_probs - y_onehot, axis=-1)
        tmp_delta_0 *= tf.reshape(self.weights,[-1])
        
        x_nextword_onehot = tf.one_hot(self.next_enc_inputs, depth=vocab_size)
        x_nextword_onehot = tf.reshape(x_nextword_onehot, [-1, vocab_size])
        
        x_nextword_onehot_reverse = tf.one_hot(self.next_enc_inputs_reverse, depth=vocab_size)
        x_nextword_onehot_reverse = tf.reshape(x_nextword_onehot_reverse, [-1, vocab_size])
        
        nextpos_onehot = tf.one_hot(self.tlm_targets_pos, depth=pos_size)
        nextpos_onehot = tf.reshape(nextpos_onehot, [-1, pos_size])
        
        nextpos_onehot_reverse = tf.one_hot(self.tlm_targets_pos_reverse, depth=pos_size)
        nextpos_onehot_reverse = tf.reshape(nextpos_onehot_reverse, [-1, pos_size])
        
        
        extra_reg_term = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]),logits=self.phi_logits)
        extra_reg_term *= tf.reshape(self.weights, [-1])
        extra_reg_term = tf.reduce_sum(extra_reg_term) / tf.to_float(self.batch_size)



        
        
        
        
        # self.loss_phi *= tf.reshape(self.weights, [-1])
        # something like this 
        loss_phi = delta(tmp_delta_0) - energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse) #+ energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse)
        loss_phi = -loss_phi
        self.loss_phi = extra_reg_term #loss_phi + 0.5 * extra_reg_term #tf.maximum(loss_phi, 0.0) + 0.5 * extra_reg_term
        
        
        lambda_new = 1.0
        new_theta_term = lambda_new * (- energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse) \
                                       + energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse))
        new_theta_term = tf.maximum(new_theta_term, -1.0)
        
        loss_theta = delta(tmp_delta_0) - energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse) \
            + energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse)
            # + 0.0001 * retrive_var_regularize(['energy_function','energy_feature_proj','energy_feature']) # regularization
        loss_theta = tf.maximum(loss_theta, -1.0)
        self.loss_theta = loss_theta + new_theta_term
        
        
        
        
        
        
        
        ''' Optimization '''
        
        phi = retrive_var(['phi_projection','phi','embedding_char'])
        theta = retrive_var(['energy_function','energy_feature','energy_feature_pos'])#,'tlm_projection','tlm','tlm_reverse'])
        self.optimizer_phi = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_phi, var_list=phi)
        self.optimizer_theta = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_theta, var_list=theta)
        
        
        psi = retrive_var(['phi_projection','phi'])
        self.loss_psi = energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse)
        self.optimizer_psi = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_psi, var_list=psi)
        

        
        
        
        
        self.saver = tf.train.Saver()
        
        








def create_model_infnet_tlm(sess, dim_h, n_tag, n_pos, n_chunk, vocab_size, load_model=False, model_path=''):
    model = InfNet_TLM(dim_h, n_tag, n_pos, n_chunk, vocab_size)
    if load_model:
        print('Loading model from ...')
        model.saver.restore(sess, model_path)
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    
    return model


