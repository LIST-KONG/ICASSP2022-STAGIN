from graphcnn.layers import *
import tensorflow as tf2
from tensorflow.python.ops.rnn import dynamic_rnn
no_ite = 0
FLAGS = tf.app.flags.FLAGS

class GraphCNNNetwork(object):
    def __init__(self):
        self.current_V = None
        self.current_A = None
        self.current_mask = None
        self.labels = None
        self.network_debug = False

    def create_network(self, input):
        self.current_V = input[0]
        self.current_A = input[1]
        self.labels = input[2]
        self.current_mask = None  # input[3]

        if self.network_debug:
            size = tf.reduce_sum(self.current_mask, axis=1)
            self.current_V = tf.Print(self.current_V,
                                      [tf.shape(self.current_V), tf.reduce_max(size), tf.reduce_mean(size)],
                                      message='Input V Shape, Max size, Avg. Size:')

        return input

    def make_graphcnn_layer(self, no_filters,  name=None, with_bn=True, with_act_func=True):#no_count, i,
        # global no_ite
        with tf.variable_scope(name, default_name='gcn') as scope:
            #################################
            # V_shape = self.current_V.get_shape()
            # reshape_V = tf.reshape(self.current_V, (-1, V_shape[2], V_shape[3]))

            # A_shape = self.current_A.get_shape()
            # result_A = tf.reshape(self.current_A, (-1, A_shape[2], A_shape[3], A_shape[4]))
            # self.current_V=make_graphcnn_layer(reshape_V, result_A,no_filters)
            ##################################
            self.current_V = make_graphcnn_layer(self.current_V, self.current_A, no_filters)
            # self.current_V = tf2.Print(self.current_V, [tf2.shape(self.current_V)], message="curren_V is the size:",summarize=4)
            if self.current_mask != None:
                self.current_mask = tf2.Print(self.current_mask, [tf2.shape(self.current_mask)],
                                              message="current_mask is the size:", summarize=4)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape()) - 1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var],
                                          message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V
    
    #gcn
    def make_trans_gcn_layer(self,no_filters, if_concat=False,name=None):
        with tf.variable_scope(name, default_name='gclstm') as scope:
            V_shape = self.current_V.get_shape()
            reshape_V = tf.reshape(self.current_V, (-1, V_shape[2], V_shape[3]))

            A_shape = self.current_A.get_shape()
            result_A = tf.reshape(self.current_A, (-1, A_shape[2], A_shape[3], A_shape[4]))

            h=make_graphcnn_layer(reshape_V, result_A,no_filters)
            h=make_bn(h, tf.constant(True, dtype=tf.bool))
            h = tf.nn.relu(h)
            self.current_V = tf.reshape(h, (-1, V_shape[1], V_shape[2], no_filters))
        return self.current_V

    #transformer
    def make_graph_transformer_layer(self,  name=None):
        with tf.variable_scope(name, default_name='transformer') as scope:
            V_shape = self.current_V.get_shape()
            ###################
            if V_shape[3]==100:
                num_heads = 10
            else:
                num_heads = 8
            ###################
            self.current_V = tf.reshape(self.current_V, tf.stack([-1, V_shape[1], V_shape[2] * V_shape[3]]))
            self.current_V=multihead_attention(queries=self.current_V,
                                              keys=self.current_V,
                                              values=self.current_V,
                                              num_heads=num_heads,
                                              dropout_rate=0.3,
                                              training=True,
                                              causality=False)
            self.current_V = tf.reshape(self.current_V, tf.stack([-1, V_shape[1], V_shape[2], V_shape[3]]))
            self.current_V = ff(self.current_V, num_units=[2048, V_shape[3]])##

        return self.current_V

    def make_batchnorm_layer(self):
        if self.current_mask != None:
            self.current_mask = tf2.Print(self.current_mask, [tf2.shape(self.current_mask)],
                                          message="current_mask is the size:", summarize=4)
        self.current_V = make_bn(self.current_V, self.is_training, mask=self.current_mask, num_updates=self.global_step)
        return self.current_V

    def make_fc_layer(self, no_filters, name=None, with_bn=False, with_act_func=True):
        with tf.variable_scope(name, default_name='FC') as scope:
            self.current_mask = None

            if len(self.current_V.get_shape()) >= 2:
                no_input_features = int(np.prod(self.current_V.get_shape()[1:]))  # change#
                # no_input_features = int(np.prod(tf.shape(self.current_V)[1:]))  # change#
                self.current_V = tf.reshape(self.current_V, [-1, no_input_features])
            self.current_V = make_embedding_layer(self.current_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V