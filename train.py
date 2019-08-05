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

parser = argparse.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--batch_size")
parser.add_argument("--init")            # 1 for Xavier, 2 for uniform random
parser.add_argument("--dropout_prob")
parser.add_argument("--decode_method")   #(0: greedy, 1: beam [default: 0])
parser.add_argument("--beam_width")
parser.add_argument("--save_dir")
parser.add_argument("--epochs")
parser.add_argument("--train")
parser.add_argument("--val")
args = parser.parse_args()

learning_rate = float(args.lr)
batch_size = int(args.batch_size)
init = int(args.init)
dropout_prob = float(args.dropout_prob)
decode_method = int(args.decode_method)
beam_width = int(args.beam_width)      
save_dir = args.save_dir
training_iters = int(args.epochs)
train = args.train
val = args.val


traindf = pd.read_csv(train, encoding = 'utf8')
train_ids = traindf.values[:,0]
tr_X = traindf.values[:,1]
tr_y = traindf.values[:,2]

train_X_flat = np.array(' '.join(tr_X).split(' '))
spch = [',','(',')','1','2','4','6','.','?','-','_','/','É','Á','È',"'"]
train_X_uni_chars = np.sort(np.array(['0','1']+[x for x in list(set(train_X_flat)) if x not in spch]))
#print(train_X_uni_chars)
train_y_flat = np.array(' '.join(tr_y).split(' '))
train_y_uni_chars = np.sort(np.array(['0','1']+[x for x in list(set(train_y_flat)) if x not in spch]))
train_X_2dlist = []
train_y_2dlist = []
for i in range(len(tr_X)):
  s1 = tr_X[i]
  s1 = s1.replace(',','')
  s1 = s1.replace('(','')
  s1 = s1.replace(')','')
  s1 = s1.replace('1','')
  s1 = s1.replace('2','')
  s1 = s1.replace('4','')
  s1 = s1.replace('6','')
  s1 = s1.replace('.','')
  s1 = s1.replace('?','')
  s1 = s1.replace("'",'')
  s1 = s1.replace(' ','')
  try:
    s1 = unicode(s1, 'utf-8')
  except NameError:
    pass
  s1 = unicodedata.normalize('NFD', s1).encode('ascii', 'ignore').decode("utf-8")
  s1 = str(s1)
  s2 = tr_y[i]
  s2 = s2.replace(',','')
  s2 = s2.replace('(','')
  s2 = s2.replace(')','')
  s2 = s2.replace('1','')
  s2 = s2.replace('2','')
  s2 = s2.replace('4','')
  s2 = s2.replace('6','')
  s2 = s2.replace('.','')
  s2 = s2.replace('?','')
  s2 = s2.replace("'",'')
  s2 = s2.replace(' ','')
  if (s1.count('_')+s1.count('-')+s1.count('/'))==(s2.count('_')+s2.count('-')+s2.count('/')):
    train_X_2dlist = train_X_2dlist + [list(x) for x in re.split("[_/-]+", s1) if len(x)>0]
    train_y_2dlist = train_y_2dlist + [list(x) for x in re.split("[_/-]+", s2) if len(x)>0]
  else:
    s1 = s1.replace('_','')
    s1 = s1.replace('-','')
    s1 = s1.replace('/','')
    s2 = s2.replace('_','')
    s2 = s2.replace('-','')
    s2 = s2.replace('/','')
    train_X_2dlist = train_X_2dlist + [list(s1)]
    train_y_2dlist = train_y_2dlist + [list(s2)]
    

# print(len(tr_X),len(train_X_2dlist))
# print(len(tr_y),len(train_y_2dlist))


train_X = [None]*len(train_X_2dlist)       #one_hot
for i in range(len(train_X_2dlist)):
  data = train_X_2dlist[i]
  label_encoder = LabelEncoder()
  label_encoder.fit(train_X_uni_chars)
  integer_encoded = label_encoder.transform(data)
  one_hot_data = np.zeros((len(integer_encoded), len(train_X_uni_chars)))
  one_hot_data[np.arange(len(integer_encoded)), integer_encoded] = 1
  train_X[i] =integer_encoded

train_y = [None]*len(train_y_2dlist)       #one_hot
for i in range(len(train_y_2dlist)):
  data = train_y_2dlist[i]
  label_encoder = LabelEncoder()
  label_encoder.fit(train_y_uni_chars)
  integer_encoded = label_encoder.transform(data)
  one_hot_data = np.zeros((len(integer_encoded), len(train_y_uni_chars)))
  one_hot_data[np.arange(len(integer_encoded)), integer_encoded] = 1
  train_y[i] =integer_encoded



validdf = pd.read_csv(val, encoding = 'utf8')
valid_ids = validdf.values[:,0]
v_X = validdf.values[:,1]
v_y = validdf.values[:,2]

valid_X_2dlist = [None]*len(v_X)
for i in range(len(v_X)):
  s = v_X[i]
  try:
    s = unicode(s, 'utf-8')
  except NameError: 
    pass
  s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("utf-8")
  v_X[i] = str(s)
  valid_X_2dlist[i] = v_X[i].split(" ")

val_X = [None]*len(valid_X_2dlist)       #one_hot
for i in range(len(valid_X_2dlist)):
  data = valid_X_2dlist[i]
  label_encoder = LabelEncoder()
  label_encoder.fit(train_X_uni_chars)
  integer_encoded = label_encoder.transform(data)
  one_hot_data = np.zeros((len(integer_encoded), len(train_X_uni_chars)))
  one_hot_data[np.arange(len(integer_encoded)), integer_encoded] = 1
  val_X[i] =integer_encoded

valid_y_2dlist = [x.split(" ") for x in v_y]
val_y = [None]*len(valid_y_2dlist)       #one_hot
for i in range(len(valid_y_2dlist)):
  data = valid_y_2dlist[i]
  label_encoder = LabelEncoder()
  label_encoder.fit(train_y_uni_chars)
  integer_encoded = label_encoder.transform(data)
  one_hot_data = np.zeros((len(integer_encoded), len(train_y_uni_chars)))
  one_hot_data[np.arange(len(integer_encoded)), integer_encoded] = 1
  val_y[i] =integer_encoded

# testdf = pd.read_csv("/content/drive/My Drive/DLPA3/test_final.csv", encoding = 'utf8')
# test_ids = testdf.values[:,0]
# te_X = testdf.values[:,1]

# test_X_2dlist = [None]*len(te_X)
# for i in range(len(te_X)):
#   s = te_X[i]
#   try:
#     s = unicode(s, 'utf-8')
#   except NameError: 
#     pass
#   s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("utf-8")
#   te_X[i] = str(s)
#   test_X_2dlist[i] = te_X[i].split(" ")

# test_X = [None]*len(test_X_2dlist)       #one_hot
# for i in range(len(test_X_2dlist)):
#   data = test_X_2dlist[i]
#   label_encoder = LabelEncoder()
#   label_encoder.fit(train_X_uni_chars)
#   integer_encoded = label_encoder.transform(data)
#   one_hot_data = np.zeros((len(integer_encoded), len(train_X_uni_chars)))
#   one_hot_data[np.arange(len(integer_encoded)), integer_encoded] = 1
#   test_X[i] =integer_encoded
# print(len(test_X))

# train_X+=val_X
# train_y+=val_y

#we give encoder input sequence like 'hello how are you', we take the last hidden state and feed to decoder and it
#will generate a decoded value. we compare that to target value, if translation would be 'bonjour ca va' and minimize 
#the difference by optimizing a loss function

#in this case we just want to encode and decode the input successfully

#bidirectional encoder
#We will teach our model to memorize and reproduce input sequence. 
#Sequences will be random, with varying length.
#Since random sequences do not contain any structure, 
#model will not be able to exploit any patterns in data. 
#It will simply encode sequence in a thought vector, then decode from it.
#this is not about prediction (end goal), it's about understanding this architecture

#this is an encoder-decoder architecture. The encoder is bidrectional so 
# #it It feeds previously generated tokens during training as inputs, instead of target sequence.
# import numpy as np #matrix math 
# import tensorflow as tf #machine learningt
# #import helpers #for formatting data into batches and generating random sequence data
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session

#First critical thing to decide: vocabulary size.
#Dynamic RNN models can be adapted to different batch sizes 
#and sequence lengths without retraining 
#(e.g. by serializing model parameters and Graph definitions via tf.train.Saver), 
#but changing vocabulary size requires retraining the model.

PAD = 0
EOS = 1

vocab_size_X = 27+2
vocab_size_y = 70+2
input_embedding_size =256 #character length
output_embedding_size =256 #character length

encoder_hidden_units = 256 #num neurons
decoder_hidden_units = encoder_hidden_units * 2 #in original paper, they used same number of neurons for both encoder
#and decoder, but we use twice as many so decoded output is different, the target value is the original input 
#in this example

batch_size = 500
#input placehodlers
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
#contains the lengths for each of the sequence in the batch, we will pad so all the same
#if you don't want to pad, check out dynamic memory networks to input variable length sequences
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_lengths = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name='decoder_lengths')
is_training =  tf.Variable(True,  name='training')
keep_prob =  tf.placeholder(shape=(None),dtype=tf.float32,  name='keep_prob')
# batch_size_pl = tf.placeholder(tf.int32, [], name='batch_size_pl')

#randomly initialized embedding matrrix that can fit input sequence
#used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
#reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
in_embeddings = tf.Variable(tf.random_uniform([vocab_size_X, input_embedding_size], -1.0, 1.0), dtype=tf.float32, name='in_embeddings')
out_embeddings = tf.Variable(tf.random_uniform([vocab_size_y, output_embedding_size], -1.0, 1.0), dtype=tf.float32, name='out_embeddings')


#this thing could get huge in a real world application
encoder_inputs_embedded = tf.nn.embedding_lookup(in_embeddings, encoder_inputs)
encoder_inputs_embedded = tf.cast(encoder_inputs_embedded,tf.float32,name='encoder_inputs_embedded') 
decoder_outputs_embedded = tf.nn.embedding_lookup(out_embeddings, decoder_targets)
decoder_outputs_embedded = tf.cast(decoder_outputs_embedded,tf.float32,name='decoder_outputs_embedded')

from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple, GRUCell, MultiRNNCell

encoder_cell = LSTMCell(encoder_hidden_units, activation = tf.tanh)
# encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell_prior, output_keep_prob=keep_prob)

#get outputs and states
#bidirectional RNN function takes a separate cell argument for 
#both the forward and backward RNN, and returns separate 
#outputs and states for both the forward and backward RNN

#When using a standard RNN to make predictions we are only taking the “past” into account. 
#For certain tasks this makes sense (e.g. predicting the next word), but for some tasks 
#it would be useful to take both the past and the future into account. Think of a tagging task, 
#like part-of-speech tagging, where we want to assign a tag to each word in a sentence. 
#Here we already know the full sequence of words, and for each word we want to take not only the 
#words to the left (past) but also the words to the right (future) into account when making a prediction. 
#Bidirectional RNNs do exactly that. A bidirectional RNN is a combination of two RNNs – one runs forward from 
#“left to right” and one runs backward from “right to left”. These are commonly used for tagging tasks, or 
#when we want to embed a sequence into a fixed-length vector (beyond the scope of this post).


((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )

encoder_fw_outputs

encoder_bw_outputs

encoder_fw_final_state

encoder_bw_final_state

#Concatenates tensors along one dimension.
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

#letters h and c are commonly used to denote "output value" and "cell state". 
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
#Those tensors represent combined internal state of the cell, and should be passed together. 

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)
# encoder_final_state = tuple(encoder_final_state)

attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=decoder_hidden_units, memory=attention_states,memory_sequence_length=encoder_inputs_length)

first_decoder_cell_prior = LSTMCell(decoder_hidden_units)
first_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(first_decoder_cell_prior, attention_mechanism, attention_layer_size=decoder_hidden_units)
second_decoder_cell_prior = LSTMCell(decoder_hidden_units)
second_decoder_cell = tf.contrib.rnn.DropoutWrapper(second_decoder_cell_prior, output_keep_prob=keep_prob)
decoder_cells = [first_decoder_cell,second_decoder_cell]
decoder_cell = MultiRNNCell(decoder_cells)

first_initial_state = first_decoder_cell.zero_state(batch_size,tf.float32)
first_initial_state = first_initial_state.clone(cell_state=encoder_final_state)
decoder_initial_state = tuple((first_initial_state,encoder_final_state))

print(encoder_outputs)

#we could print this, won't need
# encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

#decoder_lengths = encoder_inputs_length + 1
#Max length is 38
# decoder_lengths = 1
# +2 additional steps, +1 leading <EOS> token for decoder inputs

#manually specifying since we are going to implement attention details for the decoder in a sec
#weights
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size_y], -1, 1), dtype=tf.float32, name='W')
#bias
b = tf.Variable(tf.zeros([vocab_size_y]), dtype=tf.float32, name='b')

#create padded inputs for the decoder from the word embeddings

#were telling the program to test a condition, and trigger an error if the condition is false.
assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(out_embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(out_embeddings, pad_time_slice)

print()

# from tensorflow.python.layers.core import Dense
# # projection_layer = tf.layers.dense(inputs =decoder_hidden_units, units=vocab_size_y,use_bias=True)
# projection_layer = Dense(vocab_size_y)
# # Replicate encoder infos beam_width times
# beam_decoder_initial_state = tf.contrib.seq2seq.tile_batch(
#     encoder_state, multiplier=beam_width)

# # Define a beam-search decoder
# beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
#         cell=decoder_cell,
#         embedding=decoder_outputs_embedded,
#         start_tokens=start_tokens,
#         end_token=end_token,
#         initial_state=decoder_initial_state,
#         beam_width=beam_width,
#         output_layer=projection_layer,
#         length_penalty_weight=0.0,
#         coverage_penalty_weight=0.0)

# # Dynamic decoding
# outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)

# # 'state' is a tensor of shape [batch_size, cell_state_size]
# if is_training==True:
#   decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, inputs = decoder_outputs_embedded, sequence_length = decoder_lengths,
#                                    initial_state=decoder_initial_state,
#                                    dtype=tf.float32, time_major=True)
# else:
  
#   decoder_outputs, decoder_final_state = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)

#manually specifying loop function through time - to get initial cell state and input to RNN
#normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

#we define and return these values, no operations occur here
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded
    #last time steps cell state
    initial_cell_state = decoder_initial_state
    #none
    initial_cell_output = None    #none
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

#attention mechanism --choose which previously generated token to pass as input in the next timestep
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    
    def get_next_input():
        #dot product between previous ouput and weights, then + biases
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        #Logits simply means that the function operates on the unscaled output of 
        #earlier layers and that the relative scale to understand the units is linear. 
        #It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities 
        #(you might have an input of 5).
        #prediction value at current time step
        print(previous_output)
        print(output_logits)
        #Returns the index with the largest value across axes of a tensor.
        prediction = tf.argmax(output_logits, axis=1)
        print(prediction.shape)
        #embed prediction for the next input
        next_input = tf.nn.embedding_lookup(out_embeddings, prediction)
#         if is_training == True:
#           next_input_input = decoder_targets[time,:]
#           next_input_input.set_shape([batch_size])
#           next_input = tf.nn.embedding_lookup(out_embeddings, next_input_input)
#         else:
#           next_input = tf.nn.embedding_lookup(out_embeddings, prediction)
        return next_input
    
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended

    
    
    #Computes the "logical and" of elements across dimensions of a tensor.
    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    #Return either fn1() or fn2() based on the boolean predicate pred.
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    
    #set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, 
            input,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

#Creates an RNN specified by RNNCell cell and loop function loop_fn.
#This function is a more primitive version of dynamic_rnn that provides more direct access to the 
#inputs each iteration. It also provides more control over when to start and finish reading the sequence, 
#and what to emit for the output.
#ta = tensor array
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack(name='decoder_outputs')

decoder_outputs

#to convert output to human readable prediction
#we will reshape output tensor

#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#reduces dimensionality
decoder_max_steps,decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))

#flettened output tensor
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
#pass flattened tensor through decoder
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
#prediction vals
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps,decoder_batch_size, vocab_size_y))

#final prediction
decoder_prediction = tf.argmax(decoder_logits, 2,name='decoder_prediction')

#cross entropy loss
#one hot encode the target values so we don't rank just differentiate
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size_y, dtype=tf.float32),
    logits=decoder_logits,name='stepwise_cross_entropy'
)

#loss function
loss = tf.reduce_mean(stepwise_cross_entropy,name='loss')
#train it 
train_op = tf.train.AdamOptimizer().minimize(loss,name='train_op')

def createbatch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
#     print(inputs,len(inputs))
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
#           print(element,type(element))
          inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
#     print(inputs_ti#me_major,inputs_time_major.shape)
#     print(len(sequence_lengths))
    return inputs_time_major, sequence_lengths

# print(sess.run(batch_size))
batches_X = []
batches_y = []
batch_X = []
batch_y = []
excess = batch_size - len(train_X)%batch_size
for idx in range(len(train_X)):
  i = train_X[idx].tolist()
  j = train_y[idx].tolist()
  batch_X.append(i)
  batch_y.append(j)
  if (idx+1)%batch_size ==0:
    batches_X.append(batch_X)
    batches_y.append(batch_y)
    batch_X = []
    batch_y = []
batches_X.append(batch_X+[x.tolist() for x in train_X[0:excess]])
batches_y.append(batch_y+[x.tolist() for x in train_y[0:excess]])

# print(batches_X)
# batches = random_sequences(length_from=3, length_to=8,
#                                    vocab_lower=2, vocab_upper=10,
#                                    batch_size=batch_size)

def next_feed(idx):
    batch_X = batches_X[idx]
    batch_y = batches_y[idx]
    encoder_inputs_, encoder_input_lengths_ = createbatch(batch_X)
#     print(encoder_inputs_,encoder_inputs_.shape)
#     print(len(encoder_input_lengths_))
#     print(encoder_input_lengths_[0])
    decoder_targets_, decoder_lengths_ = createbatch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch_y]
    )
#     decoder_lengths = max(decoder_lengths_all)
#     print('sdfghj',decoder_lengths)
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_lengths : decoder_lengths_,
        is_training : True,
        keep_prob: 1,
    }

print(len(batches_y[0]))

pred_X = val_X
val_y
batches_X_pred = []
batch_X_pred = []
batches_y_pred = []
batch_y_pred = []
excess = batch_size - len(pred_X)%batch_size
for idx in range(len(pred_X)):
  i = pred_X[idx].tolist()
  j=val_y[idx].tolist()
  batch_X_pred.append(i)
  batch_y_pred.append(j)
  if (idx+1)%batch_size ==0:
    batches_X_pred.append(batch_X_pred)
    batch_X_pred = []
    batches_y_pred.append(batch_y_pred)
    batch_y_pred = []
batches_X_pred.append(batch_X_pred+[x.tolist() for x in pred_X[0:excess]])
batches_y_pred.append(batch_y_pred+[x.tolist() for x in val_y[0:excess]])
def next_feed_pred(idx):
    batch_X_pred = batches_X_pred[idx]
    batch_y_pred = batches_y_pred[idx]
    encoder_inputs_, encoder_input_lengths_ = createbatch(batch_X_pred)
    decoder_targets_, decoder_lengths_ = createbatch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch_y_pred]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_lengths: decoder_lengths_,
        is_training : False,
        keep_prob: 1,
    }

t_y = [' '.join(x) for x in train_y_2dlist]
def train_accuracy():
  max_batches_pred = len(batches_X)
  sumeq=0
  train_loss =0
  pred_y = []
  for batch in range(max_batches_pred):
    fd = next_feed(batch)
    fd[is_training]=False
    fd[keep_prob]=1
    batch_loss = sess.run(loss, fd)
#     print(batch_loss.shape)
    train_loss+=batch_loss
  return train_loss
# print(pred_y_str[1],v_y[1])

def val_accuracy():
  max_batches_pred = len(batches_X_pred)
  sumeq=0
  val_loss =0
  pred_y = []
  for batch in range(max_batches_pred):
    fd = next_feed_pred(batch)
    batch_loss = sess.run(loss, fd)
#     print(batch_loss.shape)
    val_loss+=batch_loss
    batch_prediction = sess.run(decoder_prediction, fd).T
    if batch == max_batches_pred-1:
      zz=0
      for l in batch_prediction:
        if zz == batch_size - excess:
          break
        pred_y.append(l)
        zz+=1
    else:
      for l in batch_prediction:
        pred_y.append(l)


  pred_y = np.asarray(pred_y)
  pred_y_str = [None]*len(pred_y)
  for i in range(len(pred_y)):
    pred_y_str[i] = " ".join(train_y_uni_chars[[x for x in pred_y[i] if x>1]])
#   print(len(pred_y_str))
  for i in range(len(v_y)):
    sumeq += (v_y[i]==pred_y_str[i])
  val_acc = sumeq/len(pred_y)
  return val_acc,val_loss
# print(pred_y_str[1],v_y[1])

def write_log_files(join_till):
    str_1 = '\n'.join(log_str_arr_train[0:join_till])
    str_2 = '\n'.join(log_str_arr_val[0:join_till])
    text_file = open('/content/drive/My Drive/DLPA3/save_test/log_train_bidirectional_again.txt', "w")
    text_file.write(str_1)
    text_file.close()
    text_file = open('/content/drive/My Drive/DLPA3/save_test/log_val_bidirectional_again.txt', "w")
    text_file.write(str_2)
    text_file.close()

nepochs = 120
es_count = 0
max_batches = len(batches_X)*nepochs
batches_in_epoch =len(batches_X)
loss_track = []
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
log_str_arr_train = ["" for x in range(nepochs)]
log_str_arr_val = ["" for x in range(nepochs)]
join_till = 0
step_counter=0
log_str_arr_indx=0
try:
    mx_acc = -1
    for batch in range(max_batches):
        fd = next_feed(batch%batches_in_epoch)
        
        _, l = sess.run([train_op, loss], fd)
        
        
        loss_track.append(l)
        if batch == 0 or (batch+1) % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(train_X_uni_chars[inp]))
                print('    predicted > {}'.format(train_y_uni_chars[pred]))
                if i >= 2:
                    break
            train_loss = train_accuracy()        
            temp_acc,temp_loss = val_accuracy()
            print(temp_acc,temp_loss)
            log_str_arr_train[log_str_arr_indx] = 'Epoch '+str(log_str_arr_indx)+', Loss: '+str(train_loss)
            log_str_arr_val[log_str_arr_indx] = 'Epoch '+str(log_str_arr_indx)+', Loss: '+str(temp_loss)
            log_str_arr_indx+=1
            if batch%3==0:
              write_log_files(log_str_arr_indx)            
            if temp_acc>=mx_acc:
              save_path = saver.save(sess, "/content/drive/My Drive/DLPA3/save_test/bidirectional_again_model_{0}{1}".format(batch,temp_acc))
              mx_acc=temp_acc
              es_count = 0
            else:
              es_count +=1
              if es_count==5:
                break
    write_log_files(log_str_arr_indx)       
except KeyboardInterrupt:
    join_till = log_str_arr_indx
    write_log_files(log_str_arr_indx)
    print('training interrupted')

