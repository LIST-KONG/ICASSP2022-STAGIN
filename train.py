from graphcnn.experiment import *
from utils import *
import tensorflow.compat.v1 as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GCNTransformerNetConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        
        net.make_trans_gcn_layer(32)   
        net.make_graph_transformer_layer()
        net.make_trans_gcn_layer(32)   
        net.make_graph_transformer_layer()  
       
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)#(bs,2)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1

def main():
    ######################
    train_batch_size = 16
    iter_time = 10
    train_iterations =800
    total_window_size=176#230
    ######################

    proportion = 0.20
    atlas = ''  
    node_number =90  
    constructor =GCNTransformerNetConstructor#GCNTransformerNetConstructor
    window_size = 100
    step = 2
    
    flag = tf.app.flags
    flag.DEFINE_integer('flag_per_sub_adj_num', 39, 'per_sub_adj_num')  # 66 39 61 56
    flag.DEFINE_integer('node_number', node_number, 'per_sub_adj_num')
   
    xinxiang_test(iter_time,train_batch_size,proportion, atlas, node_number, window_size, step, constructor, train_iterations,total_window_size)
    #修改1.fmri,length 2.flag_per_sub_adj_num
    # ASD_test_one_sample(iter_time,train_batch_size,proportion, atlas, node_number, window_size, step, constructor, train_iterations,total_window_size)
    #cahnge two dataset dir/ fold num
main()
