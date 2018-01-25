"""
Inceptionism.py
McKell Stauffer
Fall 2017
"""

import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize, imsave
from IPython.display import Image, display

if __name__ == "__main__":
    sess = tf.Session()
    opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3], dtype=tf.float32, stddev=1e-1), name='opt_img' )
    tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )
    vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

    style_img = imread( 'style.png', mode='RGB' )
    style_img = imresize( style_img, (224, 224) )
    style_img = np.reshape( style_img, [1,224,224,3] )

    content_img = imread( 'content.png', mode='RGB' )
    content_img = imresize( content_img, (224, 224) )
    content_img = np.reshape( content_img, [1,224,224,3] )

    layers = [ 'conv1_1', 'conv1_2',
               'conv2_1', 'conv2_2',
               'conv3_1', 'conv3_2', 'conv3_3',
               'conv4_1', 'conv4_2', 'conv4_3',
               'conv5_1', 'conv5_2', 'conv5_3' ]

    ops = [ getattr( vgg, x ) for x in layers ]

    content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
    style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

    w = 0.2
    alpha = 1e-3
    beta = 1

    with tf.name_scope('loss') as scope:
        cont_loss = .5 * tf.reduce_sum(tf.square(vgg.conv4_2 - content_acts[-5]))
        E = 0
        for val in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
            ind = layers.index(val)
            _, x, y, z = ops[ind].get_shape().as_list()

            layer_a = tf.reshape(style_acts[ind],[x*y, z])
            layer_g = tf.reshape(getattr(vgg, val),[x*y, z])

            A = tf.matmul(tf.transpose(layer_a), layer_a)
            G = tf.matmul(tf.transpose(layer_g), layer_g)

            E += 1/(4*(x*y*z)**2) * tf.reduce_sum(tf.square(G - A))

        sty_loss = w * E
        loss = alpha * cont_loss + beta * sty_loss

    train_step = tf.train.AdamOptimizer(.1).minimize(loss, var_list = [opt_img])

    sess.run( tf.initialize_all_variables() )
    vgg.load_weights( 'vgg16_weights.npz', sess )
    sess.run( opt_img.assign( content_img ))
    for i in range(6000):
        print(i)
        if i % 50 == 0:
            img = np.clip(img, 0, 255)
            sess.run(opt_img.assign(img))
            img = sess.run(tmp_img)
            shape = np.shape(img)
            img = np.reshape(img, [shape[1], shape[2], shape[3]])
            imsave("outfile"+str(i)+".png", img)
        ts = sess.run(train_step)
