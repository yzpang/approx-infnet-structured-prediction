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

import tensorflow.compat.v1 as tf













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


    '''

    # STEP 1 - TRAINING TAG LANGUAGE MODEL

    NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
    config = tf.ConfigProto(
        intra_op_parallelism_threads=NUM_THREADS,
        inter_op_parallelism_threads=NUM_THREADS)
    config.gpu_options.allow_growth = True

    print('configuring GPU')

    # InfNet

    steps_per_checkpoint = 1500
    learning_rate = 0.0006
    max_epochs = 1500
    batch_size = 1
    dropout = 0.7
    load_model = False # True
    model_name = './tmp/08-14-18-ner-series-ppf'


    with tf.Graph().as_default():
        
        with tf.Session(config=config) as sess:
            

            # model = create_model_infnet(sess, 100, len(tag2id)) # use 100
            model = create_model_infnet_tlm(sess, 128, len(tag2id), len(pos2id), len(chunk2id), len(word2id), load_model, model_name) # use 100
            
            if True: # training
                batches, _ = get_batches(x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id, batch_size)
                random.shuffle(batches)

                start_time = time.time()
                step = -1
                loss_tlm = 0.0
                best_dev = float('inf')
                #best_dev = 96.00

                for epoch in range(max_epochs):
                    print('----------------------------------------------------')
                    print('epoch %d, learning_rate %f' % (epoch + 1, learning_rate))

                    for batch in batches: # note
                        
                        if batch['size'] == batch_size:
                        
                            feed_dict_tmp = feed_dictionary(model, batch, dropout, learning_rate)
    #                         tmp0, tmp1 = sess.run([model.output_0_shape, model.output_1_shape],
    #                            feed_dict=feed_dict_tmp)
    #                         print(tmp0, tmp1)

                            step_loss_tlm_1, _ = sess.run([model.tlm_train_loss_1, model.optimizer_tlm_1],
                                feed_dict=feed_dict_tmp)
                            step_loss_tlm_2, _ = sess.run([model.tlm_train_loss_2, model.optimizer_tlm_2],
                                feed_dict=feed_dict_tmp)
                            step_loss_tlm = step_loss_tlm_1+step_loss_tlm_2

                            step += 1
                            loss_tlm += step_loss_tlm / steps_per_checkpoint

                            if step % steps_per_checkpoint == 0:
                                print('step %d, time %.0fs, loss_tlm %.2f' \
                                    % (step, time.time() - start_time, loss_tlm))
                                loss_tlm = 0.0
                    
                                #acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                                #print('-- dev acc: %.2f' % acc)
                
                            if step % (2*steps_per_checkpoint) == 0:
                                pp1, pp2, pp1_reverse, pp2_reverse = evaluate_tlm(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                                print('-- dev perplexity: %.2f, %.2f, %.2f, %.2f' % (pp1,pp2,pp1_reverse,pp2_reverse))
                                
                                if (3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse) < best_dev:
                                    best_dev = 3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse
                                    print('------ best dev perplexity so far -> saving model...')
                                    model.saver.save(sess, model_name)
                                
                    #pp, _ = evaluate_tlm(sess, model, x_train, y_train, pos_train, chunk_train, word2id, tag2id, pos2id, chunk2id,  batch_size)
                    #print('-- train pp: %.2f' % pp)

                    pp1, pp2, p1_reverse, pp2_reverse = evaluate_tlm(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                    print('-- dev perplexity: %.2f, %.2f, %.2f, %.2f' % (pp1,pp2,pp1_reverse,pp2_reverse))
                    
                    
                    

    # need to change dev to tweet3m_test_sent and tweet3m_test_tag
                    
                    if 3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse < best_dev:
                        best_dev = 3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse
                        print('------ best dev pp so far -> saving model...')
                        model.saver.save(sess, model_name)
                    
    #                 # acc, _ = evaluate(sess, model, test_data, window)
    #                 # print('-- test acc: %.2f' % acc)

    '''













    # NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
    # config = tf.ConfigProto(
    #     intra_op_parallelism_threads=NUM_THREADS,
    #     inter_op_parallelism_threads=NUM_THREADS)
    # config.gpu_options.allow_growth = True

    # print('configuring GPU')


    # STEP 2 - TRAINING PHI AND THETA

    # InfNet

    steps_per_checkpoint = 3000
    learning_rate = 0.0001
    max_epochs = 150
    batch_size = 1
    dropout = 0.7
    load_model = True # True
    model_name = './tmp/08-14-18-ner-series-lstm-char'

    with tf.Graph().as_default():
        with tf.Session() as sess:

            # model = create_model_infnet(sess, 100, len(tag2id)) # use 100
            model = create_model_infnet_tlm(sess, 128, len(tag2id), len(pos2id), len(chunk2id), len(word2id), load_model, model_name) # create right model!
            
            if True: # training
                batches, _ = get_batches(x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id, batch_size)
                random.shuffle(batches)

                start_time = time.time()
                step = -1
                loss_phi, loss_psi, loss_theta = 0.0, 0.0, 0.0
                #best_dev = float('-inf')
                best_dev = 95.10 # do not save

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






