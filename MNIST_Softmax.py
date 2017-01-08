import input_data
import tensorflow as tf

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

#create model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
