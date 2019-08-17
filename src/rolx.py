import math
import time
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import tensorflow as tf
from layers import Factorization
from refex import RecursiveExtractor
from print_and_read import log_setup, tab_printer, epoch_printer, log_updater, data_reader, data_saver

class Model(object):
    """
    Abstract model class.
    """
    def __init__(self,args):
        """
        Every model needs the same initialization -- args, graph.
        We delete the sampler object to save memory.
        We also build the computation graph up. 
        """
        self.args = args
        self.recurser = RecursiveExtractor(args)
        self.dataset = np.array(self.recurser.new_features)
        self.user_size = self.dataset.shape[0]
        self.feature_size = self.dataset.shape[1]
        self.nodes = range(0, self.user_size)
        self.true_step_size = (self.user_size * self.args.epochs)/self.args.batch_size
        self.build()

    def build(self):
        """
        Building the model.
        """
        pass

    def feed_dict_generator(self):
        """
        Creating the feed generator
        """
        pass

    def train(self):
        """
        Training the model.
        """
        pass
        
class ROLX(Model):
    """
    ROLX class.
    """
    def build(self):
        """
        Method to create the computational graph.
        """        
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.factorization_layer =  Factorization(self.args, self.user_size, self.feature_size)

            self.loss = self.factorization_layer()

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
            self.init = tf.global_variables_initializer()


    def feed_dict_generator(self, nodes, step):    
        """
        Method to generate left and right handside matrices, proper time index and overlap vector.
        """

        left_nodes = np.array(nodes)
        right_nodes = np.array([i for i in range(0,self.feature_size)])

        targets = self.dataset[nodes,:]

        feed_dict = {self.factorization_layer.edge_indices_left: left_nodes,
                     self.factorization_layer.edge_indices_right: right_nodes,
                     self.factorization_layer.target: targets,
                     self.step: float(step)}

        return feed_dict

    def train(self):
        """
        Method for training the embedding, logging.
        """ 
        self.current_step = 0
        self.log = log_setup(self.args)

        with tf.Session(graph = self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.args.epochs):

                random.shuffle(self.nodes)
                self.optimization_time = 0 
                self.average_loss = 0

                epoch_printer(repetition)
                for i in tqdm(range(0,len(self.nodes)/self.args.batch_size)):
                    self.current_step = self.current_step + 1
                    feed_dict = self.feed_dict_generator(self.nodes[i*self.args.batch_size:(i+1)*self.args.batch_size], self.current_step)
                    start = time.time()
                    _, loss = session.run([self.train_op , self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = (self.average_loss/i)
                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time)
                tab_printer(self.log)
            self.features = self.factorization_layer.embedding_node.eval()
            data_saver(self.features, self.args.embedding_output)
