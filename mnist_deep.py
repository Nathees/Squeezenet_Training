# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import pickle
import numpy as np 

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 64, 64, 3])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 3, 96])
    b_conv1 = bias_variable([96])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  ##-----------my net--------- fire 2
  with tf.name_scope('fire2_s11'):
    W_conv_f2_s11 = weight_variable([1, 1, 96, 16])
    b_conv_f2_s11 = bias_variable([16])
    h_conv1_f2_s11 = tf.nn.relu(conv2d(h_pool1, W_conv_f2_s11) + b_conv_f2_s11)

  with tf.name_scope('fire2_e11'):
    W_conv_f2_e11 = weight_variable([1, 1, 16, 64])
    b_conv_f2_e11 = bias_variable([64])
    h_conv1_f2_e11 = tf.nn.relu(conv2d(h_conv1_f2_s11, W_conv_f2_e11) + b_conv_f2_e11)

  with tf.name_scope('fire2_e33'):
    W_conv_f2_e33 = weight_variable([3, 3, 16, 64])
    b_conv_f2_e33 = bias_variable([64])
    h_conv1_f2_e33 = tf.nn.relu(conv2d(h_conv1_f2_s11, W_conv_f2_e33) + b_conv_f2_e33)

  h_conv1_f2 = tf.concat([h_conv1_f2_e11, h_conv1_f2_e33], 3)
  #----------------------------------------------

  ##-----------my net--------- fire 3
  with tf.name_scope('fire3_s11'):
    W_conv_f3_s11 = weight_variable([1, 1, 128, 16])
    b_conv_f3_s11 = bias_variable([16])
    h_conv1_f3_s11 = tf.nn.relu(conv2d(h_conv1_f2, W_conv_f3_s11) + b_conv_f3_s11)

  with tf.name_scope('fire3_e11'):
    W_conv_f3_e11 = weight_variable([1, 1, 16, 64])
    b_conv_f3_e11 = bias_variable([64])
    h_conv1_f3_e11 = tf.nn.relu(conv2d(h_conv1_f3_s11, W_conv_f3_e11) + b_conv_f3_e11)

  with tf.name_scope('fire3_e33'):
    W_conv_f3_e33 = weight_variable([3, 3, 16, 64])
    b_conv_f3_e33 = bias_variable([64])
    h_conv1_f3_e33 = tf.nn.relu(conv2d(h_conv1_f3_s11, W_conv_f3_e33) + b_conv_f3_e33)
  h_conv1_f3 = tf.concat([h_conv1_f3_e11, h_conv1_f3_e33], 3)
  #----------------------------------------------

  ##-----------my net--------- fire 4
  with tf.name_scope('fire4_s11'):
    W_conv_f4_s11 = weight_variable([1, 1, 128, 32])
    b_conv_f4_s11 = bias_variable([32])
    h_conv1_f4_s11 = tf.nn.relu(conv2d(h_conv1_f3, W_conv_f4_s11) + b_conv_f4_s11)

  with tf.name_scope('fire4_e11'):
    W_conv_f4_e11 = weight_variable([1, 1, 32, 128])
    b_conv_f4_e11 = bias_variable([128])
    h_conv1_f4_e11 = tf.nn.relu(conv2d(h_conv1_f4_s11, W_conv_f4_e11) + b_conv_f4_e11)

  with tf.name_scope('fire4_e33'):
    W_conv_f4_e33 = weight_variable([3, 3, 32, 128])
    b_conv_f4_e33 = bias_variable([128])
    h_conv1_f4_e33 = tf.nn.relu(conv2d(h_conv1_f4_s11, W_conv_f4_e33) + b_conv_f4_e33)
  h_conv1_f4 = tf.concat([h_conv1_f4_e11, h_conv1_f4_e33], 3)
  #----------------------------------------------

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool4'):
    h_pool1_p4 = max_pool_2x2(h_conv1_f4)
  #---------------------------------------------------

   ##-----------my net--------- fire 5
  with tf.name_scope('fire5_s11'):
    W_conv_f5_s11 = weight_variable([1, 1, 256, 32])
    b_conv_f5_s11 = bias_variable([32])
    h_conv1_f5_s11 = tf.nn.relu(conv2d(h_pool1_p4, W_conv_f5_s11) + b_conv_f5_s11)

  with tf.name_scope('fire5_e11'):
    W_conv_f5_e11 = weight_variable([1, 1, 32, 128])
    b_conv_f5_e11 = bias_variable([128])
    h_conv1_f5_e11 = tf.nn.relu(conv2d(h_conv1_f5_s11, W_conv_f5_e11) + b_conv_f5_e11)

  with tf.name_scope('fire5_e33'):
    W_conv_f5_e33 = weight_variable([3, 3, 32, 128])
    b_conv_f5_e33 = bias_variable([128])
    h_conv1_f5_e33 = tf.nn.relu(conv2d(h_conv1_f5_s11, W_conv_f5_e33) + b_conv_f5_e33)
  h_conv1_f5 = tf.concat([h_conv1_f5_e11, h_conv1_f5_e33], 3)
  #----------------------------------------------

  ##-----------my net--------- fire 6
  with tf.name_scope('fire6_s11'):
    W_conv_f6_s11 = weight_variable([1, 1, 256, 48])
    b_conv_f6_s11 = bias_variable([48])
    h_conv1_f6_s11 = tf.nn.relu(conv2d(h_conv1_f5, W_conv_f6_s11) + b_conv_f6_s11)

  with tf.name_scope('fire6_e11'):
    W_conv_f6_e11 = weight_variable([1, 1, 48, 192])
    b_conv_f6_e11 = bias_variable([192])
    h_conv1_f6_e11 = tf.nn.relu(conv2d(h_conv1_f6_s11, W_conv_f6_e11) + b_conv_f6_e11)

  with tf.name_scope('fire6_e33'):
    W_conv_f6_e33 = weight_variable([3, 3, 48, 192])
    b_conv_f6_e33 = bias_variable([192])
    h_conv1_f6_e33 = tf.nn.relu(conv2d(h_conv1_f6_s11, W_conv_f6_e33) + b_conv_f6_e33)
  h_conv1_f6 = tf.concat([h_conv1_f6_e11, h_conv1_f6_e33], 3)
#-----------------------------------------------------

  ##-----------my net--------- fire 7
  with tf.name_scope('fire7_s11'):
    W_conv_f7_s11 = weight_variable([1, 1, 384, 48])
    b_conv_f7_s11 = bias_variable([48])
    h_conv1_f7_s11 = tf.nn.relu(conv2d(h_conv1_f6, W_conv_f7_s11) + b_conv_f7_s11)

  with tf.name_scope('fire7_e11'):
    W_conv_f7_e11 = weight_variable([1, 1, 48, 192])
    b_conv_f7_e11 = bias_variable([192])
    h_conv1_f7_e11 = tf.nn.relu(conv2d(h_conv1_f7_s11, W_conv_f7_e11) + b_conv_f7_e11)

  with tf.name_scope('fire7_e33'):
    W_conv_f7_e33 = weight_variable([3, 3, 48, 192])
    b_conv_f7_e33 = bias_variable([192])
    h_conv1_f7_e33 = tf.nn.relu(conv2d(h_conv1_f7_s11, W_conv_f7_e33) + b_conv_f7_e33)
  h_conv1_f7 = tf.concat([h_conv1_f7_e11, h_conv1_f7_e33], 3)
#-----------------------------------------------------

   # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool8'):
    h_pool1_p8 = max_pool_2x2(h_conv1_f7)

  with tf.name_scope('conv8'):
    W_conv8 = weight_variable([1, 1, 384, 200])
    b_conv8 = bias_variable([200])
    h_conv8 = tf.nn.relu(conv2d(h_pool1_p8, W_conv8) + b_conv8)

  with tf.name_scope('avgpool9'):
    avg_pool = tf.nn.avg_pool(h_conv8, ksize=(1, 8, 8, 1), strides=(1, 1, 1, 1), padding='VALID')
    # avg_pool = tf.nn.pool (input =h_conv8, window_shape=[16,16], pooling_type = 'AVG', strides = [16,16], padding='SAME')

  avg_pool_sh = tf.reshape(avg_pool, [-1, 1*1*200])
  y_conv = tf.nn.softmax(avg_pool_sh)
  # # Second convolutional layer -- maps 32 feature maps to 64.
  # with tf.name_scope('conv2'):
  #   W_conv2 = weight_variable([5, 5, 32, 64])
  #   b_conv2 = bias_variable([64])
  #   h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # # Second pooling layer.
  # with tf.name_scope('pool2'):
  #   h_pool2 = max_pool_2x2(h_conv2)

  # # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # # is down to 7x7x64 feature maps -- maps this to 1024 features.
  # with tf.name_scope('fc1'):
  #   W_fc1 = weight_variable([16 * 16 * 64, 1024])
  #   b_fc1 = bias_variable([1024])

  #   h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
  #   h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  # with tf.name_scope('dropout'):
  #   keep_prob = tf.placeholder(tf.float32)
  #   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # # Map the 1024 features to 10 classes, one for each digit
  # with tf.name_scope('fc2'):
  #   W_fc2 = weight_variable([1024, 200])
  #   b_fc2 = bias_variable([200])

    # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv #, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def next_batch(num, data, labels):

    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[0:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    # labels_shuffle = np.asarray(labels_shuffle.values.reshape(len(labels_shuffle), 1))
    return data_shuffle, labels_shuffle


def main(_):
  np.set_printoptions(threshold=np.nan)
  # Import data


  # trainig data
  classes_f = open('wnids.txt', 'r')
  classes = classes_f.read()
  classes = classes.split('\n')
  input_files = []
  train_input = []
  y = []
  count = 0


  for item in classes:
    if (item !=''):
      for i in range(500):
        input_files.append('train/'+item+'/images/'+item+'_'+str(i)+'.JPEG')
        # image_raw = image_raw_f.read()
        # image_raw_f.close()
        # # ,dct_method="INTEGER_ACCURATE"
        # mage_tf_accurate = tf.image.decode_jpeg(image_raw).eval(session=sess)
        # train_input.append(mage_tf_accurate)
        # y.append(item)
    count = count + 1
    print(item + " count: " + str(count))

  tf_classes = tf.cast(classes, tf.string)

  filename_queue = tf.train.string_input_producer(input_files)
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  rgb_shape = tf.cast([64, 64, 1], tf.int32)
  my_img = tf.image.decode_jpeg(value)
  t = tf.reduce_all(tf.equal(tf.shape(my_img), rgb_shape))
  # if t is False:
  #   decoded_image = tf.image.grayscale_to_rgb(my_img)
  # else:
  #   decoded_image = my_img

  decoded_image = tf.cond(t,  lambda:tf.image.grayscale_to_rgb(my_img), lambda:my_img)
  key_str = tf.cast(key, tf.string)
  class_key = tf.string_split([key_str], '/', skip_empty=True)
  class_item = class_key.values[1] #tf.sparse_tensor_to_dense(class_key, default_value=' ')
  # class_item = class_key_dense[1]
  class_index = tf.where(tf.equal(tf_classes, class_item))
  class_index = class_index[0]
  item_one_hot = tf.one_hot(class_index, 200)
  item_one_hot_rs = tf.reshape(item_one_hot, [-1])

  image_batch, label_batch = tf.train.shuffle_batch([decoded_image, item_one_hot_rs], batch_size = 64, num_threads = 4, capacity = 100000,  min_after_dequeue = 300, shapes=[[64,64,3],[200]])

  init_op = tf.global_variables_initializer()

  rgb_train_input = []
  count = 0;
  rgb_train_input_x = []
  rgb_train_input_y = []

  # queue = tf.RandomShuffleQueue(capacity=100000,
  #                               min_after_dequeue=int(0.9*10), dtypes = [tf.string, tf.float32])#,
  #                               #shapes=source.shape, dtypes=source.dtype)
  # enqueue = queue.enqueue([key_str, tf.cast(decoded_image, tf.float32)])
  # num_threads = 4
  # qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)
  # tf.train.add_queue_runner(qr)
  # a_u = tf.placeholder(tf.float32, [None, 64,64, 3])
  # a_v = tf.placeholder(tf.string)

  # with tf.Session() as sess1:
  # 	coord = tf.train.Coordinator()
  # 	threads = tf.train.start_queue_runners(coord=coord)
  # 	sess1.run(init_op)
  	
  # 	for i in range(1000):
  # 		print(tf.shape(my_img).eval())
  # 		# print(classes.index(class_item.eval().decode('utf-8')))
  # 	coord.request_stop()
  # 	coord.join(threads)
    # Start populating the filename queue.


    # for i in range(len(input_files)): 
    #   a_v, a_u = queue.dequeue()
    #   # u = a[0][0]
    #   # a_m = tf.map_fn(lambda x : x, a)
    #   # u,v = a_m[0]
    #   print(sess.run(a_v.eval()))

      # x_i = my_img.eval()
      # y_i = key.eval()
      # y_i = str(y_i)
      # y_i = y_i.split('/')
      # y_i = y_i[1]
      # y_i = classes.index(y_i)
      # y_i = tf.one_hot(y_i, 200)
      # y_i = y_i.eval()
      # rgb_train_input_y.append(y_i)
      # if(x_i.shape == (64, 64, 1)):
      #   # print("grey scale image: " + str(count_g))
      #   x_i = tf.image.grayscale_to_rgb(x_i)
      #   x_i = x_i.eval()
      #   rgb_train_input_x.append(x_i)
      # else:
      #   rgb_train_input_x.append(x_i)      




      # print(i)

  # print(class_item_n)

  # counter = 0;
  # with tf.Session() as sess:
  #   coord = tf.train.Coordinator()
  #   threads = tf.train.start_queue_runners(coord=coord)
  #   for u, v in rgb_train_input:
  #     print('label_processing: ', counter)
  #     counter = counter + 1
  #     y_one_hot = tf.one_hot(u, 200)
  #     y_i = y_one_hot.eval()
  #     rgb_train_input_y.append(y_i)
  #     rgb_train_input_x.append(v)
  #   coord.request_stop()
  #   coord.join(threads)

  # with open('rgb_train_input_x.bin', 'wb') as fp:
  #   pickle.dump(rgb_train_input_x, fp)

  # with open('rgb_train_input_y.bin', 'wb') as fp:
  #   pickle.dump(rgb_train_input_y, fp)

  # with open('rgb_train_input_x.bin', 'rb') as fp:
  #   rgb_train_input_x = pickle.load(fp)
  #   fp.close()

  # #   pickle.dump(y, fp)
  # with open('rgb_train_input_y.bin', 'rb') as fp:
  #   rgb_train_input_y = pickle.load(fp)
  # print('rgb_train_input_y.bin')
  
  # count_g = 0
  # with tf.Session() as sess:
  #   coord = tf.train.Coordinator()
  #   threads = tf.train.start_queue_runners(coord=coord)
  #   for x_i in train_input:
  #     if(x_i.shape == (64, 64, 1)):
  #       count_g = count_g + 1
  #       print("grey scale image: " + str(count_g))
  #       x_i = tf.image.grayscale_to_rgb(x_i)
  #       rgb_train_input.append(np.asarray(x_i.eval()))
  #     else:
  #       rgb_train_input.append(x_i)


  #   coord.request_stop()
  #   coord.join(threads)
  # with open('rgb_train_input.bin', 'wb') as fp:
  #   pickle.dump(rgb_train_input, fp)
  #   fp.close()
  # exit(0)

  # print(rgb_train_input)
  # exit(0)
  # train_input = np.asarray(train_input, np.float32)
  # y = np.asarray(y, np.float32)
  # rgb_train_input = tf.convert_to_tensor(rgb_train_input[0:99])
  # y = tf.convert_to_tensor(y)
  # rgb_train_input = tf.reshape(rgb_train_input[0:99], [-1, 64*64*3])
  # rgb_train_input = tf.reshape(rgb_train_input, [-1, 64*64*3])
  # rgb_train_input = np.reshape(rgb_train_input, ())
  # batch_0, batch_1 = tf.train.batch([rgb_train_input[0:99],y[0:99]], batch_size=50, num_threads=3, capacity = 100)
  # Create the model
  # y = tf.reshape(y, [-1, 1*200])
  x = tf.placeholder(tf.float32, [None, 64,64, 3])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 200])

  # Build the graph for the deep net
  print('graph')
  y_conv = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  sum = 0;
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(100000):
      # if i % 1 == 0:
      #   train_accuracy = accuracy.eval(feed_dict={
      #       x: batch_0, y_: batch_1})
      
      # train_step.run(feed_dict={x: batch_0, y_: batch_1})

      # image_batch, label_batch = next_batch(100, rgb_train_input_x, rgb_train_input_y)
      img_bat, lab_bat = sess.run([image_batch, label_batch])
      if i % 10 == 0:
      	
      	train_accuracy = accuracy.eval(feed_dict={x: img_bat, y_: lab_bat})
      	sum = sum + train_accuracy
      	print('step %d, training accuracy %g, average acc %g' % (i, train_accuracy, sum*10/(i+1)))
      train_step.run(feed_dict={x: img_bat, y_: lab_bat})
      # n_c = i
      # if i % 1 == 0:
      #   train_accuracy = accuracy.eval(feed_dict={
      #       x: rgb_train_input[(n_c*50) % 100000 : (n_c*50 + 49)%100000], y_: y[(n_c*50) % 100000 : (n_c*50 + 49)%100000]})
      #   print('step %d, training accuracy %g' % (i, train_accuracy))
      # train_step.run(feed_dict={x: rgb_train_input[(n_c*50) % 100000 : (n_c*50 + 49)%100000], y_: y[(n_c*50) % 100000 : (n_c*50 + 49)%100000]})

    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    coord.request_stop()
    coord.join(threads)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
