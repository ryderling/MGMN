#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/6/13 22:34
# @Author   : Xiang Ling
# @Lab      : nesa.zju.edu.cn
# @File     : data.py
# ************************************

import json
import networkx as nx
import numpy as np
import os

import simgnn_utils
import utils

node_feat_name_dict = {
    'aids700nef': 'type',
    'linux': None
}

node_encoder_name_dict = {
    'aids700nef': 'OneHot',
    'linux': 'constant_1_2'
}


class GEDDataset(object):
    def __init__(self, ged_main_dir, args):
        self.args = args
        self.ged_main_dir = ged_main_dir
        data = simgnn_utils.SiameseDataSet(data_save_folder=os.path.join(self.ged_main_dir, 'data'), data_set_name=args.dataset, validation_ratio=0.25, node_feat_name=node_feat_name_dict[args.dataset],
                                           node_encoder_name=node_encoder_name_dict[args.dataset])
        distance = simgnn_utils.DistanceCalculator(root_folder=os.path.join(self.ged_main_dir, 'save'), data_name=data.data_set_name)
        
        self.data = data
        self.input_dim = data.input_dim()
        self.dist_calculator = distance
        
        self.train_val_max_number_of_nodes = data.train_val_max_number_of_nodes
        self.testing_max_number_of_nodes = data.test_max_number_of_nodes
        max_nr_nodes = max(self.train_val_max_number_of_nodes, self.testing_max_number_of_nodes)
        self.train_val_graphs = self.preprocess(data.train_val_gs, max_nr_nodes)
        self.training_graphs = self.preprocess(data.train_gs, self.data.train_val_max_number_of_nodes)
        self.val_graphs = self.preprocess(data.val_gs, self.data.train_val_max_number_of_nodes)
        self.testing_graphs = self.preprocess(data.test_gs, max_nr_nodes)
        
        true_result_test_train = simgnn_utils.load_result(dataset=data.data_set_name.lower(), model='astar', result_folder=os.path.join(self.ged_main_dir, 'result'), sim=None, sim_mat=None, dist_mat=None, row_graphs=self.testing_graphs,
                                                          col_graphs=self.train_val_graphs, scale=1.0)
        self.ground_truth = true_result_test_train.dist_norm_mat_
        self.training_triples = self._load_train_triples(dist_calculator=self.dist_calculator)
        self.validation_triples = self._load_val_train_triples(dist_calculator=self.dist_calculator)
        self.test_triples = self._load_test_triples()
    
    def extract_test_matrices(self, row):
        row_idx, col_idx = self.test_triples[row]
        batch_rows_feature = []
        batch_rows_adjacent = []
        batch_rows_mask = []
        batch_cols_feature = []
        batch_cols_adjacent = []
        batch_cols_mask = []
        for row, col in zip(row_idx, col_idx):
            row_feature, row_adj, row_mask = self.testing_graphs[row].matrices
            col_feature, col_adj, col_mask = self.train_val_graphs[col].matrices
            batch_rows_feature.append(row_feature)
            batch_rows_adjacent.append(row_adj)
            batch_rows_mask.append(row_mask)
            batch_cols_feature.append(col_feature)
            batch_cols_adjacent.append(col_adj)
            batch_cols_mask.append(col_mask)
        return np.array(batch_rows_feature), np.array(batch_rows_adjacent), np.array(batch_rows_mask), np.array(batch_cols_feature), np.array(batch_cols_adjacent), np.array(
            batch_cols_mask)
    
    def get_training_batch(self):
        
        all_feat_1 = []
        all_adj_1 = []
        all_feat_2 = []
        all_adj_2 = []
        
        all_masks_1 = []
        all_masks_2 = []
        
        ged_pairs = []
        
        for i in range(self.args.batch_size):
            m, n, norm_dist = self.training_triples.get_next_one()
            
            feature_1, adj_1, mask_1 = self.training_graphs[m].matrices
            feature_2, adj_2, mask_2 = self.training_graphs[n].matrices
            
            all_feat_1.append(feature_1)
            all_adj_1.append(adj_1)
            all_masks_1.append(mask_1)
            
            all_feat_2.append(feature_2)
            all_adj_2.append(adj_2)
            all_masks_2.append(mask_2)
            
            ged_pairs.append(norm_dist)
        
        return np.array(all_feat_1), np.array(all_adj_1), np.array(all_masks_1), np.array(all_feat_2), np.array(all_adj_2), np.array(all_masks_2), np.array(ged_pairs)
    
    def get_all_validation(self):
        
        all_feat_1 = []
        all_adj_1 = []
        all_feat_2 = []
        all_adj_2 = []
        
        all_mask_1 = []
        all_mask_2 = []
        
        ged_pair = []
        
        for i in range(len(self.validation_triples)):
            m, n, norm_dist = self.validation_triples[i]
            
            feature_1, adj_1, mask_1 = self.val_graphs[m].matrices
            feature_2, adj_2, mask_2 = self.training_graphs[n].matrices
            
            all_feat_1.append(feature_1)
            all_adj_1.append(adj_1)
            all_mask_1.append(mask_1)
            
            all_feat_2.append(feature_2)
            all_adj_2.append(adj_2)
            all_mask_2.append(mask_2)
            
            ged_pair.append(norm_dist)
        
        return np.array(all_feat_1), np.array(all_adj_1), np.array(all_mask_1), np.array(all_feat_2), np.array(all_adj_2), np.array(all_mask_2), np.array(ged_pair)
    
    def preprocess(self, gs, max_num_nodes):
        rtn = []
        for i in range(len(gs)):
            g1 = gs[i]
            feature_1, adj_1, mask_1 = self._graph_feature_adj(g1, max_num_nodes)
            g1.matrices = (feature_1, adj_1, mask_1)
            rtn.append(g1)
        return rtn
    
    def _graph_feature_adj(self, gs, max_nodes):
        
        feature = self.data.node_feat_encoder.encode(gs)
        adj = np.array(nx.to_numpy_matrix(gs))
        adj = adj + np.where(adj.transpose() > adj, 1, 0)
        adj = adj + np.eye(adj.shape[0])
        
        adj_normalized = self._dense_adj_normalization(adj)
        
        feature_padded = np.zeros((max_nodes, feature.shape[-1]))
        adj_padded = np.zeros((max_nodes, max_nodes))
        
        feature_padded[:feature.shape[0], :feature.shape[1]] = feature
        adj_padded[:adj_normalized.shape[0], :adj_normalized.shape[1]] = adj_normalized
        assert feature.shape[0] == adj_normalized.shape[0]
        masked = np.zeros(max_nodes)
        masked[:feature.shape[0]] = 1
        
        return feature_padded, adj_padded, masked
    
    @staticmethod
    def _dense_adj_normalization(adj):
        row_sum = np.array(adj.sum(1)).astype(np.float32)
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    def _load_train_triples(self, dist_calculator):
        train_length = len(self.training_graphs)
        train_pairs_triples = []
        for m in range(train_length):
            g1 = self.training_graphs[m]
            for n in range(train_length):
                g2 = self.training_graphs[n]
                _, norm_dist = dist_calculator.calculate_distance_btw_pairs(g1, g2, 1.0)
                train_pairs_triples.append((m, n, norm_dist))
        ret = simgnn_utils.SelfShuffleList(train_pairs_triples)
        return ret
    
    def _load_test_triples(self):
        row_len = len(self.testing_graphs)
        column_len = len(self.train_val_graphs)
        
        list_batches = []
        for row in range(row_len):
            batch_rows = []
            batch_cols = []
            
            for column in range(column_len):
                batch_rows.append(row)
            
            for column in range(column_len):
                batch_cols.append(column)
            
            list_batches.append((batch_rows, batch_cols))
        
        return list_batches
    
    def _load_val_train_triples(self, dist_calculator):
        val_length = len(self.val_graphs)
        train_length = len(self.training_graphs)
        val_pairs_triples = []
        for m in range(val_length):
            g1 = self.val_graphs[m]
            for n in range(train_length):
                g2 = self.training_graphs[n]
                dist, norm_dist = dist_calculator.calculate_distance_btw_pairs(g1, g2, 1.0)
                val_pairs_triples.append((m, n, norm_dist))
        
        return val_pairs_triples


class CFGDataset(object):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.func_name_dict = self.get_f_dict()
        self.gs, self.classes = self.read_graph()
        
        class_perm_path = os.path.join(self.data_dir, 'class_perm.npy')
        if os.path.isfile(class_perm_path):
            perm = np.load(class_perm_path)
        else:
            perm = np.random.permutation(len(self.classes))
            np.save(class_perm_path, perm)
        
        if len(perm) < len(self.classes):
            perm = np.random.permutation(len(self.classes))
            np.save(class_perm_path, perm)
        graphs_train, classes_train, graphs_dev, classes_dev, graphs_test, classes_test = self.partition_graph_dataset(self.gs, self.classes, [0.8, 0.1, 0.1], perm)
        self.graph_train = graphs_train
        self.classes_train = classes_train
        print("{} Train: {} graphs, {} functions".format(data_dir, len(graphs_train), len(classes_train)))
        print("{} Dev  : {} graphs, {} functions".format(data_dir, len(graphs_dev), len(classes_dev)))
        print("{} Test : {} graphs, {} functions".format(data_dir, len(graphs_test), len(classes_test)))
        
        # Fix the pairs for validation and testing
        if os.path.isfile(os.path.join(self.data_dir, 'valid.json')):
            with open(os.path.join(self.data_dir, 'valid.json')) as in_file:
                valid_ids = json.load(in_file)
            self.valid_epoch = utils.generate_epoch_pair(graphs_dev, classes_dev, batch_size, load_id=valid_ids)
        else:
            self.valid_epoch, valid_ids = utils.generate_epoch_pair(graphs_dev, classes_dev, batch_size, output_id=True)
            with open(os.path.join(self.data_dir, 'valid.json'), 'w') as out_file:
                json.dump(valid_ids, out_file)
        
        if os.path.isfile(os.path.join(self.data_dir, 'test.json')):
            with open(os.path.join(self.data_dir, 'test.json')) as in_file:
                test_ids = json.load(in_file)
            self.test_epoch = utils.generate_epoch_pair(graphs_test, classes_test, batch_size, load_id=test_ids)
        else:
            self.test_epoch, test_ids = utils.generate_epoch_pair(graphs_test, classes_test, batch_size, output_id=True)
            with open(os.path.join(self.data_dir, 'test.json'), 'w') as out_file:
                json.dump(test_ids, out_file)
    
    def get_f_dict(self):
        name_num = 0
        name_dict = {}
        for file in os.listdir(self.data_dir):
            if '.json' not in file or 'test' in file or 'valid' in file:
                continue
            f_name = os.path.join(self.data_dir, file)
            with open(f_name) as inf:
                print(f_name)
                for line in inf:
                    g_info = json.loads(line.strip())
                    if g_info['fname'] not in name_dict:
                        name_dict[g_info['fname']] = name_num
                        name_num += 1
        return name_dict
    
    def read_graph(self):
        graphs = []
        classes = []
        if self.func_name_dict is not None:
            for f in range(len(self.func_name_dict)):
                classes.append([])
        for file in os.listdir(self.data_dir):
            if '.json' not in file or 'test' in file or 'valid' in file:
                continue
            f_name = os.path.join(self.data_dir, file)
            with open(f_name) as inf:
                for line in inf:
                    g_info = json.loads(line.strip())
                    label = self.func_name_dict[g_info['fname']]
                    classes[label].append(len(graphs))
                    cur_graph = utils.graph(node_num=g_info['n_num'], label=label, name=g_info['src'])
                    for u in range(g_info['n_num']):
                        cur_graph.features[u] = np.array(g_info['features'][u])
                        for v in g_info['succs'][u]:
                            cur_graph.add_edge(u, v)
                    graphs.append(cur_graph)
        return graphs, classes
    
    @staticmethod
    def partition_graph_dataset(graphs, classes, partitions, perm):
        len_class = len(classes)
        st = 0.0
        ret = []
        for partition in partitions:
            cur_g = []
            cur_c = []
            ed = st + partition * len_class
            for cls in range(int(st), int(ed)):
                prev_class = classes[perm[cls]]
                cur_c.append([])
                for i in range(len(prev_class)):
                    cur_g.append(graphs[prev_class[i]])
                    cur_g[-1].label = len(cur_c) - 1
                    cur_c[-1].append(len(cur_g) - 1)
            ret.append(cur_g)
            ret.append(cur_c)
            st = ed
        return ret
