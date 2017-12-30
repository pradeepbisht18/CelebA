# Import the data as a numpy array
# First columns contain the name of image and 16 columns contains the Glassess status

import numpy as np
import os
import tensorflow as tf
from PIL import Image


data = np.loadtxt("Anno/list_attr_celeba.txt", dtype = str,usecols=(0,16))

data = np.delete(data, (0), axis=0)

im_x, im_y = 84, 84
no_of_images = 20000
channels = 3
def celebA_train_data():
    x_celebA = np.ndarray(shape=(1,(channels*im_x*im_y)),dtype=float)
    y_celebA = np.ndarray(shape=(1,2),dtype=float)
    count_x=0
    count_y=0
    for image_file_name in os.listdir('img_align_celeba/'):
        if image_file_name.endswith(".jpg"):
            im = Image.open('img_align_celeba/'+image_file_name)
            im=im.resize((im_x, im_y), Image.ANTIALIAS)
            x_t = np.reshape(im, [1, (channels*im_x*im_y)])
            x_celebA = np.append(x_celebA, x_t, axis=0)
            count_x = count_x +1
        if count_x == no_of_images:
            break
    for i in range(data.shape[0]):
        if int(data[i][1]) == -1:
            y_t=[0, 1]
        else:
            y_t=[1, 0]
        y_t = np.reshape(y_t, [1,2])
        y_celebA = np.append(y_celebA, y_t, axis=0)
        count_y = count_y + 1
        if count_y == no_of_images:
            break
    return x_celebA, y_celebA

x_u, y_u = celebA_train_data()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First CN Layer

x = tf.placeholder(tf.float32, [None, channels*im_x*im_y], name="Variable_input_X")
y_ = tf.placeholder(tf.float32, [None, 2], name="Variable_output_Y")

W_conv1 = weight_variable([5, 5, channels, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, im_x, im_y, channels])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second CN Layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer

W_fc1 = weight_variable([21 * 21 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 21*21*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate the model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    BATCH_SIZE = 100
    for offset in range(0, int(len(y_u)*0.9), BATCH_SIZE):
        batch_x = x_u[offset: offset + BATCH_SIZE]
        batch_y = y_u[offset: offset + BATCH_SIZE]
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        print('step %d, training accuracy: %g' %(offset, train_accuracy))
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob:0.5})
    print('Train size %d Test size %d Batch size %d'%(int(len(y_u)*0.9), int(len(y_u)*0.1), BATCH_SIZE))
    print('Test accuracy: %g' % accuracy.eval(feed_dict={x: x_u[int(len(y_u)*0.9):], y_: y_u[int(len(y_u)*0.9):], keep_prob: 1.0}))


