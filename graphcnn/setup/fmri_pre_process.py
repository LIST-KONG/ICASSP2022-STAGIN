import numpy as np
import os
import math

def max_min_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class PreProcess(object):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2,
                 data_dir='S01_mean_ts',total_widow_size=176):# 263 data_dir zoe改
        # 需要传入的参数
        self.atlas = atlas  # '' '_ho' '_bn'
        self.node_number = node_number  # 90 113 246
        self.window_size = window_size  # 60:10:100
        self.step = step  # 1 2
        self.proportion = proportion
        # 由子类初始化的数据 1. 数据类型， 2. 相应类型标签
        self.subj_type_set = []
        self.subj_label_set = []
        # 常量参数数据
        self.main_dir = '/data/ABIDE_multi/' + data_dir + '/'  # 改fMRI_preprocess_pre/
        self.total_window_size = total_widow_size   #改,total_widow_size=230 zoe改
        self.adj_number = math.ceil((self.total_window_size - self.window_size + 1) / self.step)
        # 通过PreProcess获得的中间数据
        # 1. edge_weight_matrix ：DCN邻接矩阵；
        # 2. edge_weight_matrix_binary：通过proportion得到的二值化邻接矩阵
        # 3. node_label：每个节点的标签
        self.edge_weight_matrix = []
        self.edge_weight_matrix_binary = []
        self.node_label = []
        self.gcn_node_label_matrix = []
        self.gcn_node_label_signal_matrix = []
        # gcn输入数据
        # 1. gcn_label：标签
        # 2. gcn_node_label_matrix： 节点特征
        # 3. gcn_edge_feature：邻接矩阵
        self.gcn_label = []
        self.gcn_node_feature = []
        self.gcn_edge_feature = []

    def compute_graph_cnn_input(self):
        self.compute_input_node_signal()
        # self.compute_input_node_concat()
        return np.array(self.gcn_node_feature), np.array(self.gcn_edge_feature), np.reshape(
            self.gcn_label, [-1])

    def compute_input_node_signal(self):
        self.compute_edge_weight()
        self.compute_edge_weight_binary()
        self.gcn_edge_feature = np.expand_dims(self.edge_weight_matrix_binary, axis=3)
        self.gcn_node_feature = self.gcn_node_label_signal_matrix

    def compute_input_node_sum(self):
        self.compute_edge_weight()
        self.compute_edge_weight_binary()
        # scipy.io.savemat('data.mat', {'edge_weight_matrix_binary': self.edge_weight_matrix_binary})
        self.compute_node_label()
        # 构建GCN输入
        self.compute_node_label_matrix()
        self.gcn_edge_feature = np.expand_dims(self.edge_weight_matrix_binary, axis=3)
        self.gcn_node_feature = self.gcn_node_label_matrix

    def compute_input_node_concat(self):
        self.compute_edge_weight()
        self.compute_edge_weight_binary()
        # scipy.io.savemat('data.mat', {'edge_weight_matrix_binary': self.edge_weight_matrix_binary})
        self.compute_node_label()
        self.compute_node_label_matrix()
        self.gcn_edge_feature = np.expand_dims(self.edge_weight_matrix_binary, axis=3)
        self.gcn_node_label_signal_matrix = np.array(self.gcn_node_label_signal_matrix)
        self.gcn_node_feature = np.concatenate(
            (max_min_normalization(self.gcn_node_label_signal_matrix), self.gcn_node_label_matrix), axis=-1)

    # 对某一类别的样本计算edge_weight和label
    def compute_subj_edge_weight(self, main_subj_dir, suj_label):
        subj_dir = os.listdir(main_subj_dir)
        # 对该类中每个样本计算特征
        for i in range(len(subj_dir)):
            
            # subj_file_name = main_subj_dir + '/' + subj_dir[i] + '/RegionSeries' + self.atlas + '.mat'
            # region_series = scipy.io.loadmat(subj_file_name)['RegionSeries']
            subj_file_name = main_subj_dir + '/' + subj_dir[i]# + '/RegionSeries' + self.atlas + '.mat'
            print('reading file %s'%subj_file_name)
            region_series=np.loadtxt(subj_file_name)[:,:90]
            i_subj_edge_weight = np.zeros((self.adj_number, self.node_number, self.node_number))
            i_subj_node_label_signal = np.zeros((self.adj_number, self.node_number, self.window_size))
            # 对该样本每个时间窗计算edge weight
            for j in range(0, self.total_window_size - self.window_size, self.step):
                sub_region_series = region_series[j:j + self.window_size, :]
                i_subj_node_label_signal[math.ceil(j / self.step), :, :] = np.transpose(sub_region_series)
                i_subj_edge_weight[math.ceil(j / self.step), :, :] = np.corrcoef(np.transpose(sub_region_series))
            self.edge_weight_matrix.append(i_subj_edge_weight)
            self.gcn_node_label_signal_matrix.append(i_subj_node_label_signal)
            self.gcn_label.append(suj_label)

    # 计算edge_weight_matrix和label
    def compute_edge_weight(self):
        # 对所有类别数据计算edge_weight
        for i in range(len(self.subj_type_set)):
            # 获得数据类型，标签，数据读取路径
            i_subj_type = self.subj_type_set[i]
            i_subj_label = self.subj_label_set[i]
            main_subj_dir = self.main_dir + i_subj_type
            self.compute_subj_edge_weight(main_subj_dir, i_subj_label)
        self.edge_weight_matrix = np.array(self.edge_weight_matrix)
        self.gcn_label = np.array(self.gcn_label)

    # 计算edge_weight_matrix_binary
    def compute_edge_weight_binary(self):
        self.edge_weight_matrix = np.array(self.edge_weight_matrix)
        self.edge_weight_matrix_binary = np.zeros(self.edge_weight_matrix.shape)
        # 对每个样本计算二值化邻接矩阵
        for i in range(self.edge_weight_matrix.shape[0]):
            # 对每个窗口计算二值化邻接矩阵
            for j in range(self.edge_weight_matrix.shape[1]):
                self.edge_weight_matrix_binary[i, j, :, :] = self.compute_subj_edge_weight_binary(
                    self.edge_weight_matrix[i, j, :, :])

    # 使用比例对单个邻接矩阵进行二值化计算，返回二值化矩阵
    def compute_subj_edge_weight_binary(self, subj_edge_weight):
        # 矩阵对角线置0
        edge_weight = subj_edge_weight - np.diag(np.diag(subj_edge_weight))
        # 取矩阵上三角数据
        edge_weight_list = self.mat_to_list(edge_weight)
        # 根据比例，计算需保留的个数
        reserve_num = int(round(edge_weight_list.shape[1] * self.proportion))
        # 排序
        edge_weight_list_sorted = np.sort(edge_weight_list)
        # 获取阈值
        threshold = edge_weight_list_sorted[0, -(reserve_num + 1)]
        edge_weight_binary = np.zeros((self.node_number, self.node_number))
        # 大于阈值部分置1，其余为0
        edge_weight_binary[edge_weight > threshold] = 1
        # scipy.io.savemat("data.mat", {'edge_weight_binary': edge_weight_binary})
        return edge_weight_binary

    # 取矩阵上三角，并转换为一维向量
    def mat_to_list(self, graph_matrix):
        graph_list = np.zeros((1, int(self.node_number * (self.node_number - 1) / 2)))
        index_start = 0
        for i in range(self.node_number - 1):
            index_end = index_start + self.node_number - i - 1
            graph_list[0, index_start: index_end] = graph_matrix[i, (i + 1): self.node_number]
            index_start = index_end
        # graph_list[0, index_start] = graph_matrix[self.node_number - 2, self.node_number - 1]
        return graph_list

    # 根据二值化邻接矩阵计算节点标签
    def compute_node_label(self):
        self.edge_weight_matrix_binary = np.array(self.edge_weight_matrix_binary)
        self.node_label = np.zeros(self.edge_weight_matrix_binary.shape[0:3])
        # 对每个样本计算二值化邻接矩阵
        for i in range(self.edge_weight_matrix_binary.shape[0]):
            # 对每个窗口计算二值化邻接矩阵
            for j in range(self.edge_weight_matrix_binary.shape[1]):
                self.node_label[i, j, :] = np.sum(self.edge_weight_matrix_binary[i, j, :, :], axis=1)

    # 根据节点标签，构建GCN输入的节点特征
    def compute_node_label_matrix(self):
        self.node_label = np.array(self.node_label)
        max_node_label = 0
        # 计算node_label最大值
        for i in range(self.node_label.shape[0]):
            for j in range(self.node_label.shape[1]):
                max_node_label = max(max_node_label, max(self.node_label[i, j, :]))
        self.gcn_node_label_matrix = np.zeros(np.append(self.node_label.shape[0:3], int(max_node_label)))
        # 构建GCN输入的节点特征
        for i in range(self.node_label.shape[0]):
            for j in range(self.node_label.shape[1]):
                for k in range(int(max_node_label)):
                    # 若node_label[i,j]中某节点值为k+1, 则gcn_node_label_matrix[i,j]的相应节点的第k维值置1。
                    self.gcn_node_label_matrix[i, j, :, k] = np.equal(self.node_label[i, j, :], k + 1).astype(
                        np.float32)

class XinXiangHCMDDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2,
                 data_dir='xinxiang_data_dti',total_window_size=230):
        super(XinXiangHCMDDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir,total_window_size)
        self.subj_type_set = ['HC', 'MDD']
        self.subj_label_set = [0, 1]

class ASDPreProcess(PreProcess):
    def __init__(self, atlas='', node_number=90, window_size=100, step=1, proportion=0.2,
                 data_dir='',total_window_size=230):
        super(ASDPreProcess, self).__init__(atlas, node_number, window_size, step, proportion, data_dir,total_window_size)
        self.subj_type_set = ['HC', 'ASD']
        self.subj_label_set = [0, 1]

# test = RDNRDPreProcess()
# dataset = test.compute_graph_cnn_input()
# print('end')
