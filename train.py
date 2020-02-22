#import progressbar
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
#import matplotlib.pyplot as plt

from utils import *
from evaluate import *
from infnet import *

import tensorflow.compat.v1 as tf













if __name__ == '__main__':
    args = load_arguments()

    print('loaded arguments')


    train_data = load_sent(args.train)
    dev_data = load_sent(args.dev)
    test_data = load_sent(args.test)

    embedding2id, id2embedding, embedding_old = load_embedding(args.embedding) # change embedding_global

    tag2id, id2tag = convert_tag_to_id(train_data+dev_data+test_data, 3)
    pos2id, id2pos = convert_tag_to_id(train_data+dev_data+test_data, 1)
    chunk2id, id2chunk = convert_tag_to_id(train_data+dev_data+test_data, 2)

    train_data = preprocess_data_according_to_rules(train_data, embedding_old)
    dev_data = preprocess_data_according_to_rules(dev_data, embedding_old)
    test_data = preprocess_data_according_to_rules(test_data, embedding_old)

    word2id, id2word = construct_word_id(train_data+dev_data+test_data) 

    embedding_global = construct_embedding(len(word2id), 100, embedding_old, word2id)

    x_train, y_train, pos_train, chunk_train, case_train, num_train = turn_data_into_x_y(train_data, word2id)
    x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev = turn_data_into_x_y(dev_data, word2id)
    x_test, y_test, pos_test, chunk_test, case_test, num_test = turn_data_into_x_y(test_data, word2id)





    all_chars = ['<padunk>']+list(string.punctuation+string.ascii_uppercase+string.ascii_lowercase+string.digits)
    id2char = all_chars
    char2id = {}
    for x in all_chars:
        char2id[x] = all_chars.index(x)
    embedding_char_global = construct_embedding_char()



    x_train_full = x_train
    y_train_full = y_train







    # STEP 1 - TRAINING TAG LANGUAGE MODEL


    # NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
    # config = tf.ConfigProto(
    #     intra_op_parallelism_threads=NUM_THREADS,
    #     inter_op_parallelism_threads=NUM_THREADS)
    # config.gpu_options.allow_growth = True

    # print('configuring GPU')


    # STEP 2 - TRAINING PHI AND THETA

    # InfNet

    steps_per_checkpoint = args.steps_per_checkpoint # default 3000
    learning_rate = args.learning_rate # default 0.0003; should halve learning rate if no increase for 5 epochs
    max_epochs = args.max_epochs 
    batch_size = args.batch_size # 1 works the best for now
    dropout = args.dropout
    load_model = args.load_model
    model_name = args.model_path

    dim_h = args.dim_h

    with tf.Graph().as_default():
        with tf.Session() as sess:

            # model = create_model_infnet(sess, 100, len(tag2id)) # use 100
            model = create_model_infnet_tlm(sess, dim_h, len(tag2id), len(pos2id), len(chunk2id), len(word2id), load_model, model_name) # create right model!
            
            if True: # training
                batches, _ = get_batches(x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id, batch_size)
                random.shuffle(batches)

                start_time = time.time()
                step = -1
                loss_phi, loss_psi, loss_theta = 0.0, 0.0, 0.0
                best_dev = float('-inf')
                # best_dev = 60.00 # do not save below this number

                for epoch in range(max_epochs):
                    print('----------------------------------------------------')
                    print('epoch %d, learning_rate %f' % (epoch + 1, learning_rate))

                    for batch in batches:
                        
                        if batch['size'] == batch_size:
                                                    
                            feed_dict_tmp = feed_dictionary(model, batch, dropout, learning_rate)
                            
    #                         ### debug
    #                         tmp0, tmp1 = sess.run([model.shape0, model.shape1],
    #                            feed_dict=feed_dict_tmp)
    #                         print(tmp0, tmp1)
    #                         ###
                            

                            step_loss_phi, _ = sess.run([model.loss_phi, model.optimizer_phi],
                                feed_dict=feed_dict_tmp)
    #                         step_loss_psi, _ = sess.run([model.loss_psi, model.optimizer_psi],
    #                             feed_dict=feed_dict_tmp)
    #                         step_loss_theta, _ = sess.run([model.loss_theta, model.optimizer_theta],
    #                             feed_dict=feed_dict_tmp)

                            step += 1
                            loss_phi += step_loss_phi / steps_per_checkpoint
                            loss_psi += 0 #step_loss_psi / steps_per_checkpoint
                            loss_theta += 0 #step_loss_theta / steps_per_checkpoint

                            if step % steps_per_checkpoint == 0:
                                print('step %d, time %.0fs, loss_phi %.2f, loss_psi %.2f, loss_theta %.2f' \
                                    % (step, time.time() - start_time, loss_phi, loss_psi, loss_theta))
                                loss_phi, loss_psi, loss_theta = 0.0, 0.0, 0.0
                                
                    
    #                print('------ ... -> saving model...')
    #                model.saver.save(sess, model_name)
                    
                                #acc, _ = evaluate(sess, model, x_dev, y_dev, word2id, tag2id, batch_size)
                                #print('-- dev acc: %.2f' % acc)
                
                            if step % (2*steps_per_checkpoint) == 0: # MODIFY LATER
                                acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                                print('-- dev acc: %.2f' % acc)






                                acc, probs_test, batches_test, acc_y_test, acc_y_hat_test = evaluate_print(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id, batch_size)
                                #acc, probs_test, batches_test, acc_y_test, acc_y_hat_test = evaluate_print(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                                #print('-- dev acc: %.2f' % acc)

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
                                    print("F1 test:")
                                    print(float(tmp[1][-6:-1]))










                                
                                if acc > best_dev:
                                    best_dev = acc
                                    print('------ best dev acc so far -> saving model...')
                                    model.saver.save(sess, model_name)
                                    
                                    acc, _ = evaluate(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id,  batch_size)
                                    print('-- test acc: %.2f' % acc)
                                    
                                
                    acc, _ = evaluate(sess, model, x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id,  batch_size)
                    print('-- train acc: %.2f' % acc)

                    acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id,  batch_size)
                    print('-- dev acc: %.2f' % acc)
                    

                    if acc > best_dev:
                        best_dev = acc
                        print('------ best dev acc so far -> saving model...')
                        model.saver.save(sess, model_name)
                        
                        acc, _ = evaluate(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id,  batch_size)
                        print('-- test acc: %.2f' % acc)
                    

                    if int(time.time()-start_time) > 13000:
                        print('=== saving model after 13000 seconds -> saving')
                        model.saver.save(sess, model_name+'-contd')






