#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/3/4 14:21
# @Author   : Xiang Ling
# @Lab      : nesa.zju.edu.cn
# @File     : utils.py
# ************************************

import networkx as nx
import numpy as np
import os
from scipy import stats


def metrics_spearmanr_rho(true, predication):
    assert true.shape == predication.shape
    rho, p_val = stats.spearmanr(true, predication)
    return rho


def metrics_kendall_tau(true, predication):
    assert true.shape == predication.shape
    tau, p_val = stats.kendalltau(true, predication)
    return tau


def metrics_mean_square_error(true, predication):
    assert true.shape == predication.shape
    mse = (np.square(true - predication).mean())
    return mse


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 'Make dirs of # {} '.format(directory)
    else:
        return "the dirs already exist! Cannot be created"


def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def print_args(args, file_path):
    d = max(map(len, args.__dict__.keys())) + 1
    with open(file_path, 'w') as f:
        for k, v, in args.__dict__.items():
            f.write(k.ljust(d) + ': ' + str(v) + '\n')


def read_all_gexf_graphs(dir):
    """
    read all the files with .gexf to networkx graph
    :param dir:
    :return: list of graphs
    """
    graphs = []
    
    for file in os.listdir(dir):
        if file.rsplit('.')[-1] != 'gexf':
            continue
        file_path = os.path.join(dir, file)
        g = nx.readwrite.gexf.read_gexf(file_path)
        graphs.append(g)
    
    return graphs


class graph(object):
    def __init__(self, node_num=0, label=None, name=None, prefix_name_label=None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.prefix_name_label = prefix_name_label
        self.features = []  # node feature matrix
        self.succs = []
        self.preds = []
        if node_num > 0:
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
    
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)


def generate_epoch_pair(graphs, classes, batch, output_id=False, load_id=None):
    epoch_data = []
    id_data = []  # [ ([(G0,G1),(G0,G1), ...], [(G0,H0),(G0,H0), ...]), ... ]
    
    if load_id is None:
        st = 0
        while st < len(graphs):
            if output_id:
                input1, input2, adj1, adj2, y, pos_id, neg_id = get_pair(graphs, classes, batch, st=st, output_id=True)
                id_data.append((pos_id, neg_id))
            else:
                input1, input2, adj1, adj2, y = get_pair(graphs, classes, batch, st=st)
            epoch_data.append((input1, input2, adj1, adj2, y))
            st += batch
    else:  # Load from previous id_data
        id_data = load_id
        for id_pair in id_data:
            input1, input2, adj1, adj2, y = get_pair(graphs, classes, batch, load_id=id_pair)
            epoch_data.append((input1, input2, adj1, adj2, y))
    
    if output_id:
        return epoch_data, id_data
    else:
        return epoch_data


def get_pair(graphs, classes, batch, st=-1, output_id=False, load_id=None, output_each_label=False):
    if load_id is None:
        len_class = len(classes)
        
        if st + batch > len(graphs):
            batch = len(graphs) - st
        ed = st + batch
        
        pos_ids = []  # [(G_0, G_1), ... ]
        neg_ids = []  # [(G_0, H_0), ... ]
        
        for g_id in range(st, ed):
            g0 = graphs[g_id]
            cls = g0.label  # function name label index of graph
            tot_g = len(classes[cls])
            
            # positive pair
            if len(classes[cls]) >= 2:
                g1_id = classes[cls][np.random.randint(tot_g)]
                while g_id == g1_id:
                    g1_id = classes[cls][np.random.randint(tot_g)]
                pos_ids.append((g_id, g1_id))
            else:
                pos_ids.append((g_id, g_id))
            
            # negative pair
            cls2 = np.random.randint(len_class)
            while (len(classes[cls2]) == 0) or (cls2 == cls):
                cls2 = np.random.randint(len_class)
            tot_g2 = len(classes[cls2])
            g2_id = classes[cls2][np.random.randint(tot_g2)]
            neg_ids.append((g_id, g2_id))
    
    else:
        pos_ids = load_id[0]
        neg_ids = load_id[1]
    
    batch_pos = len(pos_ids)
    batch_neg = len(neg_ids)
    batch = batch_pos + batch_neg
    
    max_num_1 = 0
    max_num_2 = 0
    for pair in pos_ids:
        max_num_1 = max(max_num_1, graphs[pair[0]].node_num)
        max_num_2 = max(max_num_2, graphs[pair[1]].node_num)
    for pair in neg_ids:
        max_num_1 = max(max_num_1, graphs[pair[0]].node_num)
        max_num_2 = max(max_num_2, graphs[pair[1]].node_num)
    
    feature_dim = len(graphs[0].features[0])
    x1_input = np.zeros((batch, max_num_1, feature_dim))
    x2_input = np.zeros((batch, max_num_2, feature_dim))
    adj1 = np.zeros((batch, max_num_1, max_num_1))
    adj2 = np.zeros((batch, max_num_2, max_num_2))
    y_input = np.zeros(batch)
    
    # if output_each_label:
    x1_labels = np.zeros(batch)
    x2_labels = np.zeros(batch)
    
    for i in range(batch_pos):
        y_input[i] = 1
        g1 = graphs[pos_ids[i][0]]
        g2 = graphs[pos_ids[i][1]]
        for u in range(g1.node_num):
            x1_input[i, u, :] = np.array(g1.features[u])
            for v in g1.succs[u]:
                adj1[i, u, v] = 1
        for u in range(g2.node_num):
            x2_input[i, u, :] = np.array(g2.features[u])
            for v in g2.succs[u]:
                adj2[i, u, v] = 1
        if output_each_label:
            x1_labels[i] = g1.prefix_name_label  # TODO: label or prefix label
            x2_labels[i] = g2.prefix_name_label
    
    for i in range(batch_pos, batch_pos + batch_neg):
        y_input[i] = -1
        g1 = graphs[neg_ids[i - batch_pos][0]]
        g2 = graphs[neg_ids[i - batch_pos][1]]
        for u in range(g1.node_num):
            x1_input[i, u, :] = np.array(g1.features[u])
            for v in g1.succs[u]:
                adj1[i, u, v] = 1
        for u in range(g2.node_num):
            x2_input[i, u, :] = np.array(g2.features[u])
            for v in g2.succs[u]:
                adj2[i, u, v] = 1
        if output_each_label:
            x1_labels[i] = g1.prefix_name_label  # TODO: label or prefix label
            x2_labels[i] = g2.prefix_name_label  # TODO: label or prefix label
    
    if not output_each_label:
        if output_id:
            return x1_input, x2_input, adj1, adj2, y_input, pos_ids, neg_ids
        else:
            return x1_input, x2_input, adj1, adj2, y_input
    else:
        if output_id:
            return x1_input, x2_input, adj1, adj2, y_input, x1_labels, x2_labels, pos_ids, neg_ids
        else:
            return x1_input, x2_input, adj1, adj2, y_input, x1_labels, x2_labels
