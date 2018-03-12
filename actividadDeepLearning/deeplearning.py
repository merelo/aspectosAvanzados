# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:54:01 2018

@author: Miguel Merelo
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
batch_size = 128
test_size = 256
img_size = 28
num_classes = 10
def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
        conv1 = tf.nn.conv2d(X, w,strides=[1, 1, 1, 1],padding='SAME')
        conv1_a = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        conv1 = tf.nn.dropout(conv1, p_keep_conv)
        conv2 = tf.nn.conv2d(conv1, w2,strides=[1, 1, 1, 1],padding='SAME')
        conv2_a = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        conv2 = tf.nn.dropout(conv2, p_keep_conv)
        conv3=tf.nn.conv2d(conv2, w3,strides=[1, 1, 1, 1],padding='SAME')
        conv3 = tf.nn.relu(conv3)
        FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        FC_layer = tf.reshape(FC_layer, [-1, w4.get_shape().as_list()[0]])
        FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)
        output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
        output_layer = tf.nn.dropout(output_layer, p_keep_hidden)
        result = tf.matmul(output_layer, w_o)
        return result

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([28, 28], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, 28, 28, 1))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, img_size, img_size, 1)
teX = teX.reshape(-1, img_size, img_size, 1)



trXScale=central_scale_images(trX,[1.2,1.1,0.9,0.8])
trX=tf.concat([trX,trXScale],0)
trY=tf.concat([trY,np.repeat(trY,4,axis=0)],0)
del(trXScale)
trX=tf.Session().run(trX)
trY=tf.Session().run(trY)

#for i in range(28):
#    for a in range(28):
#        print("*" if trX[54999,i,a]>0 else ' ',end="")
#    print()
#print(trY[54999])


    
# 28x28x1 input img
# 28x28x1 input img
X = tf.placeholder("float", [None, img_size, img_size, 1])
Y = tf.placeholder("float", [None, num_classes])
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 *4 * 4, 625])
w_o = init_weights([625,num_classes])
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
cost = tf.reduce_mean(Y_)
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
with tf.Session() as sess:
        init_g = tf.global_variables_initializer()
        sess.run(init_g)
        for i in range(100):
                training_batch =zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
                for start, end in training_batch:
                        sess.run(optimizer , feed_dict={X: trX[start:end], Y: trY[start:end],
                                p_keep_conv: 0.8, p_keep_hidden: 0.5})
                test_indices = np.arange(len(teX)) # Get A Test Batch
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:test_size]
                print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                sess.run
                (predict_op,
                feed_dict={X: teX[test_indices],
                Y: teY[test_indices],
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0})))
