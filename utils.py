import graphcnn.setup.fmri_pre_process as fmri_pre_proc
import numpy as np
from graphcnn.experiment import *

def xinxiang_test(iter_time,train_batch_size,proportion, atlas, node_number, window_size, step, constructor, train_iterations,total_window_size):
    input_data_process = fmri_pre_proc.XinXiangHCMDDPreProcess(proportion=proportion, atlas=atlas, node_number=node_number,
                                                       window_size=window_size, step=step, data_dir='xinxiang_data_fmri',total_window_size=total_window_size)
    data_set = input_data_process.compute_graph_cnn_input()
    random_s = np.array([125, 125, 125, 125, 125, 125, 125, 125, 125,125], dtype=int)
    run_experiment(iter_time,train_batch_size,data_set, constructor, node_number, proportion, train_iterations, random_s,'HC_MDD')


def ASD_test_one_sample(iter_time,train_batch_size,proportion, atlas, node_number, window_size, step, constructor, train_iterations,total_window_size):
    input_data_process = fmri_pre_proc.ASDPreProcess(proportion=proportion, atlas=atlas, node_number=node_number,
                                                       window_size=window_size, step=step, data_dir='ABIDE_all',total_window_size=total_window_size)
    data_set = input_data_process.compute_graph_cnn_input()
    # random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)
    random_s = np.array([125, 125, 125, 125, 125, 125, 125, 125, 125,125], dtype=int)
    run_experiment(iter_time,train_batch_size,data_set, constructor, node_number, proportion, train_iterations,random_s, 'HC_ASD',)

def run_experiment(iter_time,train_batch_size,data_set, constructor, node_number, proportion, train_iterations, random_s,name):
    acc_set = np.zeros((iter_time, 1))
    std_set = np.zeros((iter_time, 1))
    sen_set = np.zeros((iter_time, 1))
    sen_std_set = np.zeros((iter_time, 1))
    spe_set = np.zeros((iter_time, 1))
    spe_std_set = np.zeros((iter_time, 1))
    f1_set = np.zeros((iter_time, 1))
    f1_std_set = np.zeros((iter_time, 1))
    auc_set = np.zeros((iter_time, 1))
    auc_std_set = np.zeros((iter_time, 1))
    attr_set = []
    for iter_num in range(iter_time):
        # Decay value for BatchNorm layers, seems to work better with 0.3
        GraphCNNGlobal.BN_DECAY = 0.3

        exp = GraphCNNExperiment('NC_ASD', 'sct_transformer', constructor())#

        exp.num_iterations = train_iterations
        exp.train_batch_size = train_batch_size
        exp.optimizer = 'adam'
        exp.debug = True##TrueFalse

        exp.preprocess_data(data_set)
        acc, std, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity,mean_f1_score,std_f1_score,mean_auc,std_auc = exp.run_kfold_experiments(#, attr
            no_folds=10,random_state=random_s[iter_num])#
        print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))
        print_ext('sensitivity is: %.2f (+- %.2f)' % (mean_sensitivity, std_sensitivity))
        print_ext('specificity is: %.2f (+- %.2f)' % (mean_specificity, std_specificity))
        print_ext('f1_score is: %.2f (+- %.2f)' % (mean_f1_score, std_f1_score))
        print_ext('auc is: %.2f (+- %.2f)' % (mean_auc, std_auc))
        acc_set[iter_num] = acc
        std_set[iter_num] = std
        sen_set[iter_num] = mean_sensitivity
        sen_std_set[iter_num] = std_sensitivity
        spe_set[iter_num] = mean_specificity
        spe_std_set[iter_num] = std_specificity
        f1_set[iter_num] = mean_f1_score
        f1_std_set[iter_num] = std_f1_score
        auc_set[iter_num] = mean_auc
        auc_std_set[iter_num] = std_auc
        # attr_set.append(attr)

    # attr_set = np.array(attr_set)
    # path = 'results/' + name + '.mat'
    # scipy.io.savemat(path, {'attr_set': attr_set})
    acc_mean = np.mean(acc_set)
    acc_std = np.std(acc_set)
    sen_mean = np.mean(sen_set)
    sen_std = np.std(sen_set)
    spe_mean = np.mean(spe_set)
    spe_std = np.std(spe_set)
    f1_mean = np.mean(f1_set)
    f1_std = np.std(f1_set)
    auc_mean = np.mean(auc_set)
    auc_std = np.std(auc_set)
    print_ext('finish!')
    verify_dir_exists('./results/')
    with open('./results/ASD.txt', 'a+') as file:
        for iter_num in range(iter_time):
            print_ext('acc %d :    %.2f   sen :    %.2f   spe :    %.2f   f1 :    %.2f   auc :    %.2f' % (
                iter_num, acc_set[iter_num], sen_set[iter_num], spe_set[iter_num],f1_set[iter_num],auc_set[iter_num]))
            file.write('%s\tacc %d :   \t%.2f (+- %.2f)\tsen :   \t%.2f (+- %.2f)\tspe :   \t%.2f (+- %.2f)f1 :   \t%.2f (+- %.2f)auc :   \t%.2f (+- %.2f)\n' % (
                str(datetime.now()), iter_num, acc_set[iter_num], std_set[iter_num], sen_set[iter_num],
                sen_std_set[iter_num], spe_set[iter_num], spe_std_set[iter_num],f1_set[iter_num], f1_std_set[iter_num],auc_set[iter_num], auc_std_set[iter_num]))
        print_ext('acc:     %.2f(+-%.2f)   sen:     %.2f(+-%.2f)   spe:     %.2f(+-%.2f)   f1:     %.2f(+-%.2f)   auc:     %.2f(+-%.2f)v' % (
            acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std,f1_mean, f1_std,auc_mean, auc_std))
        file.write('%s\t %.2f acc  :   \t%.2f (+- %.2f)  sen  :   \t%.2f (+- %.2f)  spe  :   \t%.2f (+- %.2f)  f1  :   \t%.2f (+- %.2f)  auc  :   \t%.2f (+- %.2f)\n' % (
            str(datetime.now()), proportion, acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std, f1_mean, f1_std, auc_mean, auc_std))