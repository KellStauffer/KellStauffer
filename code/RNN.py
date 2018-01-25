"""
RNN.py
McKell Stauffer
Fall 2017
"""

import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
import random
from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import RNNCell
import subprocess

class mygru( RNNCell ):
    def __init__( self, num_units1, vocab_size1 ):
        self.num_units = num_units1
        self.vocab = vocab_size1
    @property
    def state_size(self):
        return self.num_units
    @property
    def output_size(self):
        return self.vocab
    def __call__( self, inputs, state, scope=None):
        with tf.name_scope("call") as scope:
            batch_size, input_size = inputs.get_shape().as_list()
            W1 = tf.get_variable("W1", [input_size, self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            W2 = tf.get_variable("W2", [input_size, self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            W3 = tf.get_variable("W3", [input_size, self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            U1 = tf.get_variable("U1", [self.num_units, self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            U2 = tf.get_variable("U2", [self.num_units, self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            U3 = tf.get_variable("U3", [self.num_units, self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            b1 = tf.get_variable("b1", [self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            b2 = tf.get_variable("b2", [self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            b3 = tf.get_variable("b3", [self.num_units], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
            z = tf.sigmoid(tf.nn.bias_add(tf.matmul(inputs, W1) + tf.matmul(state, U1), b1))
            r = tf.sigmoid(tf.nn.bias_add(tf.matmul(inputs, W2) + tf.matmul(state, U2), b2))
            h = tf.multiply(z, state) + tf.multiply((1 - z), tf.tanh(tf.nn.bias_add(tf.matmul(inputs, W3) + tf.matmul(tf.multiply(r, state), U3), b3)))
            return h, h

def sample( num=200, prime='ab'):
    s_state = sess.run( s_initial_state )
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )
        retval = sess.run( ops, feed_dict=feed )
        s_probsv = retval[0]
        s_state = retval[1:]
        sample = np.argmax( s_probsv[0] )
        pred = data_loader.chars[sample]
        ret += pred
        char = pred
    return ret

if __name__ == "__main__":
    batch_size = 100
    sequence_length = 50
    data_loader = TextLoader(".", batch_size, sequence_length)
    vocab_size = data_loader.vocab_size
    state_dim = 128
    num_layers = 2
    tf.reset_default_graph()
    in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
    targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
    in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )
    inputs = tf.split( in_onehot, sequence_length, axis=1 )
    inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
    targets = tf.split( targ_ph, sequence_length, axis=1 )
    with tf.variable_scope('Computation_Graph') as scope:
        cell1 = mygru(state_dim, vocab_size)
        cell2 = mygru(state_dim, vocab_size)
        mcell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
        initial_state = mcell.zero_state(batch_size, dtype=tf.float32)
        output, final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, mcell)
        W = tf.Variable(tf.random_normal([state_dim, vocab_size], stddev=0.02))
        b = tf.Variable(tf.random_normal([vocab_size], stddev=0.01))
        logits = [tf.matmul(o, W) + b for o in output]
        weights = [1.0 for _ in range(sequence_length)]
        loss = tf.contrib.legacy_seq2seq.sequence_loss(logits, targets, weights)
    with tf.name_scope('Optimizer') as scope:
        optim =  tf.train.AdamOptimizer(1e-3).minimize(loss)
    batch_size = 1
    sequence_length = 1
    s_in_ph = tf.placeholder( tf.int32, [ batch_size], name='sinputs' )
    s_in_onehot = tf.one_hot( s_in_ph, vocab_size, name="input_onehot" )
    with tf.variable_scope('Sampler') as scope:
            s_initial_state = mcell.zero_state(batch_size, dtype=tf.float32)
            output1, s_final_state = tf.contrib.legacy_seq2seq.rnn_decoder([s_in_onehot], s_initial_state, mcell) # Does output need a new name
            s_probs = [tf.matmul(o, W) + b for o in output1]
    sess = tf.Session()
    sess.run( tf.global_variables_initializer() )
    summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )
    lts = []
    print("FOUND %d BATCHES" % data_loader.num_batches)
    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()
    for i in range( data_loader.num_batches ):
        x,y = data_loader.next_batch()
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]
        ops = [optim, loss]
        ops.extend( list(final_state) )
        retval = sess.run( ops, feed_dict=feed )
        lt = retval[1]
        state = retval[2:]
        print(sample( num=60, prime="a" ))
    summary_writer.close()
