# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#%matplotlib inline
import os
import pandas as pd
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
import random
import argparse
import sys
from sklearn.preprocessing import LabelEncoder
import re
import unicodedata

tf.reset_default_graph()
batch_size=500
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('/content/drive/My Drive/DLPA3/save_test/model_30790.39418254764292876.meta')
  g = tf.get_default_graph()
  new_saver.restore(sess, '/content/drive/My Drive/DLPA3/save_test/model_30790.39418254764292876')
  print([x for x in g.get_operations()])
  encoder_inputs = g.get_tensor_by_name('encoder_inputs:0')
  encoder_inputs_length = g.get_tensor_by_name('encoder_inputs_length:0')
  decoder_lengths = g.get_tensor_by_name('decoder_lengths:0')
  decoder_prediction = g.get_tensor_by_name('decoder_prediction:0')
  is_training = g.get_tensor_by_name('training:0')
  keep_prob = g.get_tensor_by_name('keep_prob:0')
  batches_X_test = []
  batch_X_test = []
  pred_X = test_X
  excess = batch_size - len(test_X)%batch_size
  for idx in range(len(pred_X)):
    i = test_X[idx].tolist()
    batch_X_test.append(i)
    if (idx+1)%batch_size ==0:
      batches_X_test.append(batch_X_test)
      batch_X_test = []
  print(excess)
  batches_X_test.append(batch_X_test+[x.tolist() for x in test_X[0:excess]])
  print()
  def next_feed_test(idx):
      batch_X_test = batches_X_test[idx]
      encoder_inputs_, encoder_input_lengths_ = createbatch(batch_X_test)

      return {
          encoder_inputs: encoder_inputs_,
          encoder_inputs_length: encoder_input_lengths_,
          decoder_lengths: [x * 2 for x in encoder_input_lengths_],
          is_training : False,
          keep_prob: 1,
      }
  max_batches_test = len(batches_X_test)
  loss_track = []
  test_y=[]

  for batch in range(max_batches_test):
      fd = next_feed_test(batch)
      test_y.append((sess.run(decoder_prediction, fd).T))
  print(test_y[0].shape[0])
  # matras = train_y_uni_chars[2:5].tolist()+train_y_uni_chars[49:63].tolist()+[train_y_uni_chars[71]]
  # nonmatras = [item for item in train_y_uni_chars.tolist() if item not in matras]
  test_y_str = [None]*len(test_X)
  print(len(test_X))
  for batch in range(max_batches_test):
    for i in range(test_y[batch].shape[0]):
      if batch_size*batch+i == len(test_X):
        break


      test_y_str[batch_size*batch+i] = " ".join(train_y_uni_chars[[x for x in test_y[batch][i,:] if x>1]])
  import csv
  predicted_output = open('/content/drive/My Drive/DLPA3/predicted_output_2l.csv','w')
  predicted_output.write('id,HIN\n')
  for i in range(len(test_y_str)):
    predicted_output.write(str(i)+','+test_y_str[i]+'\n')
  predicted_output.close()