"""
GAN.py
McKell Stauffer
Fall 2017
"""

import tensorflow as tf
import numpy as np
import os
from skimage import io, transform
from matplotlib import pyplot as plt
import random

def generator(x, name = "gen", R=True):
    with tf.variable_scope("Generator") as scope:
        x = tf.reshape(x, [1, int(np.prod(x.shape))])
        g0 = tf.layers.dense(x, 4*4*1024, name="g0", reuse = R, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        g0 = tf.reshape(g0, [1, 4, 4, 1024])
        g1 = tf.layers.conv2d_transpose(g0, 512, 5, 2, padding='SAME', reuse = R, activation = tf.nn.relu, name="g1", kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        g2 = tf.layers.conv2d_transpose(g1, 256, 5, 2, padding='SAME', reuse = R, activation = tf.nn.relu, name="g2", kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        g3 = tf.layers.conv2d_transpose(g2, 128, 5, 2, padding='SAME', reuse = R, activation = tf.nn.relu, name="g3", kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        g4 = tf.layers.conv2d_transpose(g3, 3, 5, 2, padding='SAME', reuse = R, activation = tf.tanh, name="g4", kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        return (g4+1)/2

def discriminator(x, name = "dis", R=True):
    with tf.variable_scope("Discriminator") as scope:
        d0 = tf.layers.conv2d(x, 3, 5, 2, padding = "SAME", activation = lrelu, name = "d0", reuse = R, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        d1 = tf.layers.conv2d(d0, 128, 5, 2, padding = "SAME", activation = lrelu, name = "d1", reuse = R, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        d2 = tf.layers.conv2d(d0, 256, 5, 2, padding = "SAME", activation = lrelu, name = "d2", reuse = R, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        d3 = tf.layers.conv2d(d0, 512, 5, 2, padding = "SAME", activation = lrelu, name = "d3", reuse = R, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        d3 = tf.reshape(d3, [1, int(np.prod(d3.shape))])
        d4 = tf.layers.dense(d3, 1, name="d4", reuse=R, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        return d4

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

if __name__ == "__main__":
    files = os.listdir('img_align_celeba/img_align_celeba')
    lamb = 10
    n = 10
    alph = 0.0001
    b1 = 0.5
    b2 = 0.999

    tf.reset_default_graph()

    input_data = tf.placeholder(tf.float32, [1, 64, 64, 3])

    with tf.name_scope("set_up") as scope:
        ru = tf.random_uniform([100], 0, 1)
        x = generator(ru, R= False)
        D_fake = discriminator(x, R = False)

    with tf.name_scope("train_gen") as scope:
        ru1 = tf.random_uniform([100], 0, 1)
        x1 = generator(ru1)
        D_fake1 = discriminator(x1)
        Lg = -D_fake1[0][0]

    with tf.name_scope("wgan_gp") as scope:
        ru2 = tf.random_uniform([100], 0, 1)
        x2 = generator(ru2)
        x2n = alph*input_data+(1-alph)*x2
        D_fake2 = discriminator(x2)
        D_real = discriminator(input_data)
        D_intrp = discriminator(x2n)
        grad = tf.gradients(D_intrp, [x2n])[0]
        slope = tf.norm(grad)
        Ld = (D_fake2 - D_real + lamb*(slope-1)**2)[0][0]

    with tf.name_scope("loss") as scope:
        lst_vars = tf.trainable_variables()
        gen_var_lst = [var for var in lst_vars if var.name.startswith('Generator')]
        dis_var_lst = [var for var in lst_vars if var.name.startswith('Discriminator')]
        train_gen = tf.train.AdamOptimizer(alph, b1, b2).minimize(Lg, var_list = gen_var_lst)
        train_dis = tf.train.AdamOptimizer(alph, b1, b2).minimize(Ld, var_list = dis_var_lst)

    init=tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess=tf.Session()
    sess.run(init)

    j = 0
    for i in range(2):
        for f in files:
            j += 1
            feat = transform.resize(io.imread('img_align_celeba/img_align_celeba/' + f), (64, 64, 3), mode='constant')
            feat = np.reshape(feat, (1, 64, 64, 3))
            for _ in range(n):
                sess.run(train_dis, feed_dict={input_data: feat})
            tg = sess.run([train_gen, x1, Lg, Ld], feed_dict={input_data: feat})
            if j % 500 == 0:
                x_val = sess.run(x1)
                x_val = np.reshape(x_val, (64, 64, 3))
                name = "images/name"+str(j)+".jpg"
                scipy.misc.toimage(x_val).save(name)
                saver.save(sess, 'checkpoints/my-model'+str(j)+'.ckpt')
