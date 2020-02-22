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
                            step_loss_psi, _ = sess.run([model.loss_psi, model.optimizer_psi],
                                feed_dict=feed_dict_tmp)
                            step_loss_theta, _ = sess.run([model.loss_theta, model.optimizer_theta],
                                feed_dict=feed_dict_tmp)

                            step += 1
                            loss_phi += step_loss_phi / steps_per_checkpoint
                            loss_psi += step_loss_psi / steps_per_checkpoint
                            loss_theta += step_loss_theta / steps_per_checkpoint

                            if step % steps_per_checkpoint == 0:
                                print('step %d, time %.0fs, loss_phi %.2f, loss_psi %.2f, loss_theta %.2f' \
                                    % (step, time.time() - start_time, loss_phi, loss_psi, loss_theta))
                                loss_phi, loss_psi, loss_theta = 0.0, 0.0, 0.0
                                
                                acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                                print('-- dev acc: %.2f' % acc)



                                acc_dev, probs_dev, batches_dev, acc_y_dev, acc_y_hat_dev = evaluate_print(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                                print('-- dev acc: %.2f' % acc_dev)
                                f1_dev = compute_f1(probs_dev, batches_dev, acc_y_dev, acc_y_hat_dev)

                                
                                if f1_dev > best_dev:
                                    best_dev = f1_dev
                                    print('------ best dev acc so far -> saving model...')
                                    model.saver.save(sess, model_name)
                                    
                                    # can skip                         
                                    acc_test, probs_test, batches_test, acc_y_test, acc_y_hat_test = evaluate_print(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id, batch_size)
                                    print('-- test acc: %.2f' % acc_test)
                                    f1_test = compute_f1(probs_test, batches_test, acc_y_test, acc_y_hat_test)
                                    print('-- test f1: %.2f' % f1_test)
                                    
                                
                    acc, _ = evaluate(sess, model, x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id,  batch_size)
                    print('-- train acc: %.2f' % acc)

                    # acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id,  batch_size)
                    # print('-- dev acc: %.2f' % acc)
                    acc_dev, probs_dev, batches_dev, acc_y_dev, acc_y_hat_dev = evaluate_print(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                    print('-- dev acc: %.2f' % acc_dev)
                    f1_dev = compute_f1(probs_dev, batches_dev, acc_y_dev, acc_y_hat_dev)
                    print('-- dev f1: %.2f' % f1_dev)


                    if f1_dev > best_dev:
                        best_dev = f1_dev
                        print('------ best dev f1 so far -> saving model...')
                        model.saver.save(sess, model_name)

                        # can skip
                        acc_test, probs_test, batches_test, acc_y_test, acc_y_hat_test = evaluate_print(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id, batch_size)
                        print('-- test acc: %.2f' % acc_test)
                        f1_test = compute_f1(probs_test, batches_test, acc_y_test, acc_y_hat_test)
                        print('-- test f1: %.2f' % f1_test)
                    

                    # suppose slurm has strict time limit; generally not needed
                    # if int(time.time()-start_time) > 13000:
                    #     print('=== saving model after 13000 seconds -> saving')
                    #     model.saver.save(sess, model_name+'-contd')

