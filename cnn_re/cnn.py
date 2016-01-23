__author__ = 'hadyelsahar'


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf


class CNN(BaseEstimator, ClassifierMixin):

    @staticmethod
    def weight_variable(shape, name):
        """
        To create this model, we're going to need to create a lot of weights and biases.
        One should generally initialize weights with a small amount of noise for symmetry breaking,
        and to prevent 0 gradients. Since we're using ReLU neurons,
        it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons.
        " Instead of doing this repeatedly while we build the model,
        let's create two handy functions to do it for us.
        :param shape:
        :return:
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, validate_shape=False, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, validate_shape=False, name=name)

    @staticmethod
    def conv2d_valid(x, W):
        # by choosing [1,1,1,1] and "same" the output dimension == input dimension
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    @staticmethod
    def conv2d_same(x, W):
        # by choosing [1,1,1,1] and "same" the output dimension == input dimension
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def max_pool_16x1(x):
        return tf.nn.max_pool(x, ksize=[1, 16, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')


    def __init__(self, input_shape, classes, iterations, batch_size, log_dir='./log/'):
        #self.net_work_origin(input_shape, classes, iterations, batch_size)
        self.net_work_diy(input_shape, classes, iterations, batch_size, log_dir=log_dir)

    def net_work_diy(self, input_shape, classes, iterations, batch_size, log_dir):
        # assume x : [ batch x 20 x 320 x c ]
        self.m, self.n, self.c = input_shape
        self.classes = classes

        self.iterations = iterations
        self.batch_size = batch_size
        self.dropout = 0.5

        self.sess = tf.InteractiveSession()

        # 4 dimensional   datasize x seqwidth x veclength x channels
        self.x = tf.placeholder("float",  [None, self.m, self.n, self.c], name="x-input")
        self.y_ = tf.placeholder("float", [None, len(self.classes)], name="y-input")        # 2dimensional    datasize x class

        W_conv1 = CNN.weight_variable([5, self.n, self.c, 150], name="w_conv1")
        b_conv1 = CNN.bias_variable([150], name="b_conv1")

     
        with tf.name_scope("conv_1") as scope:
            h_conv1 = tf.nn.relu(CNN.conv2d_valid(self.x, W_conv1) + b_conv1)
            h_relu1 = tf.nn.relu(h_conv1)
            h_pool1 = CNN.max_pool_16x1(h_relu1)

        with tf.name_scope("fully_connected") as scope:
            self.keep_prob = tf.placeholder("float")
            h_fc1_drop = tf.nn.dropout(h_pool1, self.keep_prob)
            h_fc1_drop_flat = tf.reshape(h_fc1_drop, [-1, 150])

            W_fc1 = CNN.weight_variable([150, len(self.classes)], name="w_fc1")
            b_fc1 = CNN.bias_variable([len(self.classes)], name="b_fc1")

            h_fc1 = tf.matmul(h_fc1_drop_flat, W_fc1) + b_fc1

            self.y_conv = tf.nn.softmax(h_fc1)

        # Add summary ops to collect data
        _ = tf.histogram_summary("weights", W_conv1)
        _ = tf.histogram_summary("biases", b_conv1)
        _ = tf.histogram_summary("y", self.y_conv)

        with tf.name_scope("xent") as scope:
            cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))

        with tf.name_scope("train") as scope:
            #1e-6
            self.train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

        with tf.name_scope("test") as scope:
            self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
            self.accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)

        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph_def)
        """   correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
        """

    def fit(self, X, y, test_interval, save_path='', snapshot=500):
        self.sess.run(tf.initialize_all_variables())
        _, indices = np.unique(y, return_inverse=True)
        self.m = X.shape[1]
        self.n = X.shape[2]

        # change y from class id into array 1 hot vector
        # eg  id = 7   -->   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        temp = np.zeros((len(indices), len(self.classes)), np.int)
        for c, i in enumerate(indices):
            temp[c][i] = 1
        y = temp

        data = Batcher(X, y, self.batch_size)

        for i in range(self.iterations):
            batch = data.next_batch()

            if i % test_interval == 0:
                # debug_here()
                #train_accuracy = self.accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                #print("step %d, training accuracy %g" % (i, train_accuracy))

                feed = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}
                result = self.sess.run([self.merged, self.accuracy], feed_dict=feed)
                summary_str = result[0]
                print summary_str
                acc = result[1]
                self.writer.add_summary(summary_str, i)
                print("step %d, training accuracy %s" % (i, acc))
            else:
                feed = {self.x: batch[0], self.y_: batch[1], self.keep_prob: self.dropout}
                self.sess.run(self.train_step, feed_dict=feed)
                # self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: self.dropout})

            if save_path and ((i % snapshot == 0) or (i == self.iterations - 1)):
                self.save('{}/iters-{}.model'.format(save_path, i))


        return self

    def predict(self, X):
        """
        :param X:   4d tensor of inputs to predict [d, m, n, c]
                    d : the size of the training data
                    m : number of inputs  layer0 (+ padding)
                    n : size of each vector representation of each input to layer 0
                    c : number of input channels  (3 for images rgb,  1 or more for text)
        :return:
        """

        y_prop = self.y_conv.eval(feed_dict={self.x: X, self.keep_prob: 1.0})
        y_pred = tf.argmax(y_prop, 1).eval()
        return y_pred

    # not tested yet
    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=save_path)
        print("Model saved in file: %s" % save_path)

    # not tested yet
    def restore(self, model_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

    def net_work_origin(self, input_shape, classes, iterations, batch_size, dropout=0.5):
        conv_shape = [5, 25]
        self.m, self.n, self.c = input_shape
        self.conv_w, self.conv_l = conv_shape
        self.classes = classes

        self.iterations = iterations
        self.batch_size = batch_size
        self.dropout = dropout

        # 4 dimensional   datasize x seqwidth x veclength x channels
        self.x = tf.placeholder("float",  [None, self.m, self.n, self.c])
        self.y_ = tf.placeholder("float", [None, len(self.classes)])        # 2dimensional    datasize x class

        self.conv_height, self.conv_width = conv_shape

        W_conv1 = CNN.weight_variable([self.conv_height, self.conv_width, 1, 32])
        b_conv1 = CNN.bias_variable([32])

        h_conv1 = tf.nn.relu(CNN.conv2d(self.x, W_conv1) + b_conv1)
        h_pool1 = CNN.max_pool_2x2(h_conv1)

        W_conv2 = CNN.weight_variable([3, 3, 32, 64])
        b_conv2 = CNN.bias_variable([64])

        h_conv2 = tf.nn.relu(CNN.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = CNN.max_pool_2x2(h_conv2)

        # calculating shape of h_pool2
        # conv2d with our conf. keeps original size
        # max pooling : reduces size into half
        h_pool2_l = np.ceil(np.ceil(self.m/2.0)/2.0)
        h_pool2_w = np.ceil(np.ceil(self.n/2.0)/2.0)
        h_pool2_flat_shape = np.int32(h_pool2_l * h_pool2_w * 64)

        W_fc1 = CNN.weight_variable([h_pool2_flat_shape, 1024])
        b_fc1 = CNN.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_flat_shape])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = CNN.weight_variable([1024, len(self.classes)])
        b_fc2 = CNN.bias_variable([len(self.classes)])

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        #1e-6
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        self.sess = tf.InteractiveSession()
    
class Batcher:
    """
    a helper class to create batches given a dataset
    """
    def __init__(self, X, y, batch_size=50):
        """
        :param X: array(any) : array of whole training inputs
        :param y: array(any) : array of correct training labels
        :param batch_size: integer : default = 50,
        :return: self
        """
        self.X = X
        self.y = y
        self.iterator = 0
        self.batch_size = batch_size

    def next_batch(self):
        """
        return the next training batch
        :return: the next batch inform of a tuple (input, label)
        """
        start = self.iterator
        end = self.iterator+self.batch_size
        self.iterator = end if end < len(self.X) else 0
        return self.X[start:end], self.y[start:end]


