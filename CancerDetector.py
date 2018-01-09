"""
CancerDetector.py
McKell Stauffer
Fall 2017
"""

import tensorflow as tf
import numpy as np
import os
from skimage import io, transform
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import random
import matplotlib.image as mpimg

def conv( x, filter_size=3, stride=1, num_filters=2, is_output=False, name="conv" ):
    x_shape = x.get_shape().as_list()
    with tf.name_scope(name) as scope:
        W = tf.get_variable(name+"Filter", [filter_size, filter_size, x_shape[-1], num_filters], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
        B = tf.get_variable(name+"bias", [num_filters], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
        x = tf.nn.conv2d(x, W, [1, stride, stride, 1], "SAME")
        x = tf.nn.bias_add(x, B)
        if not is_output:
            x = tf.nn.relu(x)
        return x

def without_generalizer(mu, std):
    tf.reset_default_graph()

    input_data = tf.placeholder(tf.float32, [1, 512, 512, 3] )
    y_true = tf.placeholder(tf.int64, [512, 512] )
    y_hat = tf.placeholder(tf.float32, [512, 512])

    with tf.name_scope("DNN") as scope:
        h0 = conv(input_data, name = "h0")
        h1 = conv(h0, name = "h1")
        h2 = conv(h1, name = "h2")
        h3 = conv(h2, name = "h3", is_output = True)

    with tf.name_scope("loss_function") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_true, [1, 512, 512]), logits = h3)

    with tf.name_scope("accuracy") as scope:
        y_hat = tf.argmax(h3, axis = 3)
        correct_prediction = tf.equal(y_true, y_hat)
        accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    train_accuracy = []
    test_accuracy = []
    gen_error = []
    trd_files = os.listdir('cancer_data/inputs/train')
    ted_files = os.listdir('cancer_data/inputs/test')

    train = random.sample(trd_files, 100)
    test = random.sample(ted_files, 20)


    for i in range(10):
        train_acc = 0
        for file in train:
            feat = transform.resize(io.imread('cancer_data/inputs/train/' + file), (512, 512, 3), mode='constant')
            feat = np.reshape(feat, (1, 512, 512, 3))
            feat -= mu
            feat /= std
            label = np.array(transform.resize(io.imread('cancer_data/outputs/train/' + file), (512,512,3), mode='constant', order=0))[:,:,1]
            label = np.reshape(label, (512, 512))
            acc, ts = sess.run([accuracy, train_step], feed_dict={input_data: feat, y_true: label})
            train_acc += acc
        train_accuracy.append(train_acc)
        test_acc = 0
        for file in test:
            feat = transform.resize(io.imread('cancer_data/inputs/test/' + file), (512, 512, 3), mode='constant')
            feat = np.reshape(feat, (1, 512, 512, 3))
            feat -= mu
            feat /= std
            label = np.array(transform.resize(io.imread('cancer_data/outputs/test/' + file), (512, 512, 3), mode='constant', order=0))[:,:,1]
            label = np.reshape(label, (512, 512))
            acc = sess.run([accuracy], feed_dict={input_data: feat, y_true: label})
            test_acc += acc[0]
        test_acc = test_acc*5
        test_accuracy.append(test_acc)
        gen_error.append(np.abs(train_acc-test_acc))
        print(train_acc, test_acc)

def dropout(mu, std):
    tf.reset_default_graph()

    input_data = tf.placeholder(tf.float32, [1, 512, 512, 3] )
    y_true = tf.placeholder(tf.int64, [512, 512] )
    y_hat = tf.placeholder(tf.float32, [512, 512])

    with tf.name_scope("DNN") as scope:
        h0 = conv(input_data, name = "h0")
        h1 = conv(h0, name = "h1")
        d0 = tf.nn.dropout(h1, .1, name = "d0")
        h2 = conv(d0, name = "h2")
        h3 = conv(h2, name = "h3", is_output = True)

    with tf.name_scope("loss_function") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_true, [1, 512, 512]), logits = h3)

    with tf.name_scope("accuracy") as scope:
        y_hat = tf.argmax(h3, axis = 3)
        correct_prediction = tf.equal(y_true, y_hat)
        accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    train_accuracy = []
    test_accuracy = []
    gen_error = []
    trd_files = os.listdir('cancer_data/inputs/train')
    ted_files = os.listdir('cancer_data/inputs/test')

    train = random.sample(trd_files, 100)
    test = random.sample(ted_files, 20)

    for i in range(10):
        train_acc = 0
        for file in train:
            feat = transform.resize(io.imread('cancer_data/inputs/train/' + file), (512, 512, 3), mode='constant')
            feat = np.reshape(feat, (1, 512, 512, 3))
            feat -= mu
            feat /= std
            label = np.array(transform.resize(io.imread('cancer_data/outputs/train/' + file), (512,512,3), mode='constant', order=0))[:,:,1]
            label = np.reshape(label, (512, 512))
            acc, ts = sess.run([accuracy, train_step], feed_dict={input_data: feat, y_true: label})
            train_acc += acc
        train_accuracy.append(train_acc)
        test_acc = 0
        for file in test:
            feat = transform.resize(io.imread('cancer_data/inputs/test/' + file), (512, 512, 3), mode='constant')
            feat = np.reshape(feat, (1, 512, 512, 3))
            feat -= mu
            feat /= std
            label = np.array(transform.resize(io.imread('cancer_data/outputs/test/' + file), (512, 512, 3), mode='constant', order=0))[:,:,1]
            label = np.reshape(label, (512, 512))
            acc = sess.run([accuracy], feed_dict={input_data: feat, y_true: label})
            test_acc += acc[0]
        test_acc *= 5
        test_accuracy.append(test_acc)
        print(train_acc, test_acc)
        gen_error.append(np.abs(train_acc-test_acc))

def L1_regularizer(mu, std):
    tf.reset_default_graph()

    input_data = tf.placeholder(tf.float32, [1, 512, 512, 3] )
    y_true = tf.placeholder(tf.int64, [512, 512] )
    y_hat = tf.placeholder(tf.float32, [512, 512])

    with tf.name_scope("DNN") as scope:
        h0 = conv(input_data, name = "h0")
        h1 = conv(h0, name = "h1")
        h2 = conv(h1, name = "h2")
        h3 = conv(h2, name = "h3", is_output = True)

    with tf.name_scope("loss_function") as scope:
        l1reg = tf.contrib.layers.l1_regularizer(0.01, scope=None)
        weights = weights = tf.trainable_variables()
        reg_pen = tf.contrib.layers.apply_regularization(l1reg, weights)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_true, [1, 512, 512]), logits = h3) + reg_pen

    with tf.name_scope("accuracy") as scope:
        y_hat = tf.argmax(h3, axis = 3)
        correct_prediction = tf.equal(y_true, y_hat)
        accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    train_accuracy = []
    test_accuracy = []
    gen_error = []
    trd_files = os.listdir('cancer_data/inputs/train')
    ted_files = os.listdir('cancer_data/inputs/test')

    train = random.sample(trd_files, 100)
    test = random.sample(ted_files, 20)

    for i in range(100):
        train_acc = 0
        for file in train:
            feat = transform.resize(io.imread('cancer_data/inputs/train/' + file), (512, 512, 3), mode='constant')
            feat = np.reshape(feat, (1, 512, 512, 3))
            feat -= mu
            feat /= std
            label = np.array(transform.resize(io.imread('cancer_data/outputs/train/' + file), (512,512,3), mode='constant', order=0))[:,:,1]
            label = np.reshape(label, (512, 512))
            acc, ts = sess.run([accuracy, train_step], feed_dict={input_data: feat, y_true: label})
            train_acc += acc
        train_accuracy.append(train_acc)
        test_acc = 0
        for file in test:
            feat = transform.resize(io.imread('cancer_data/inputs/test/' + file), (512, 512, 3), mode='constant')
            feat = np.reshape(feat, (1, 512, 512, 3))
            feat -= mu
            feat /= std
            label = np.array(transform.resize(io.imread('cancer_data/outputs/test/' + file), (512, 512, 3), mode='constant', order=0))[:,:,1]
            label = np.reshape(label, (512, 512))
            acc = sess.run([accuracy], feed_dict={input_data: feat, y_true: label})
            test_acc += acc[0]
        test_acc *= 5
        test_accuracy.append(test_acc)
        gen_error.append(np.abs(train_acc-test_acc))
        print(train_acc, test_acc)

if __name__ == "__main__":
    trd_files = os.listdir('cancer_data/inputs/train')
    ted_files = os.listdir('cancer_data/inputs/test')
    values = []
    for file in trd_files[:500]:
        values.append(transform.resize(io.imread('cancer_data/inputs/train/' + file), (512, 512, 3), mode='constant'))
    for file in ted_files[:75]:
        values.append(transform.resize(io.imread('cancer_data/inputs/test/' + file), (512, 512, 3), mode='constant'))
    mu = np.mean(values)
    std = np.std(values)
    values = []
    # without_generalizer(mu, std)
    # dropout(mu, std)
    # L1_regularizer(mu, std)
