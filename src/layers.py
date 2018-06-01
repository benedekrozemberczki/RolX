import tensorflow as tf
import math
import numpy as np

class Factorization:
    """
    Factorization layer class.
    """
    def __init__(self, args, user_size, feature_size):
        """
        Initialization of the layer with proper matrices and biases.
        The input variables are also initialized here.
        """
        self.args = args
        self.user_size = user_size
        self.feature_size = feature_size 

        self.target = tf.placeholder(tf.int64, shape=[None,None])
        self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
        self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])


        self.embedding_node = tf.Variable(tf.random_uniform([self.user_size, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))

        self.embedding_feature = tf.Variable(tf.random_uniform([self.feature_size, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))
      

    def __call__(self):
        """
        Calculating the predictive loss.
        """
        self.embedding_left = tf.nn.embedding_lookup(self.embedding_node , self.edge_indices_left) 
        self.embedding_right = tf.nn.embedding_lookup(self.embedding_feature, self.edge_indices_right)

        self.embedding_predictions = tf.sigmoid(tf.matmul(self.embedding_left, tf.transpose(self.embedding_right)))

        return tf.reduce_mean(tf.losses.log_loss(self.target,self.embedding_predictions))
