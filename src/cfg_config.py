#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/7/20 15:59
# @Author   : Xiang Ling
# @Lab      : nesa.zju.edu.cn
# @File     : cfg_config.py
# ************************************

import argparse

parser = argparse.ArgumentParser(description="Multi-Level Graph Matching Network for the Graph-Graph Classification tasks")

parser.add_argument('--data_dir', type=str, default='../data/CFG', help='root directory for the data set')
parser.add_argument('--dataset', type=str, default="ffmpeg", help='indicate the specific data set (ffmpeg/OpenSSL)')
parser.add_argument('--graph_size_min', type=int, default=3, help='min node size for one graph ')
parser.add_argument('--graph_size_max', type=int, default=200, help='max node size for one graph ')
parser.add_argument('--graph_init_dim', type=int, default=6, help='init feature dimension for one graph')

parser.add_argument("--task", type=str, default='classification', help="classification/regression")

parser.add_argument("--filters", type=str, default='100_100_100', help="filters (neurons) for graph neural networks")
parser.add_argument("--conv", type=str, default='gcn', help="one kind of graph neural networks")
parser.add_argument("--match", type=str, default='node-graph', help="indicating the matching method")
parser.add_argument("--perspectives", type=int, default=100, help='number of perspectives for node-graph matching')
parser.add_argument("--match_agg", type=str, default='bilstm', help="aggregator")
parser.add_argument("--hidden_size", type=int, default=100, help='hidden size for the graph-level embedding')

# global-level information
parser.add_argument("--global_flag", type=lambda x: (str(x).lower() == 'true'), default='True', help="Whether use global info ")
parser.add_argument("--global_agg", type=str, default='fc_max_pool', help="aggregation function for global level gcn ")

# training parameters for classification tasks
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

parser.add_argument("--batch_size", type=int, default=5, help="Number of graph pairs per batch.")
parser.add_argument("--lr", type=float, default=0.5e-3, help="Learning rate.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")

# others
parser.add_argument('--gpu_index', type=str, default='1', help="gpu index to use")
parser.add_argument('--log_path', type=str, default='../CFGLogs/', help='path for log file')
parser.add_argument('--repeat_run', type=int, default=1, help='indicated the index of repeat run')

# only test
parser.add_argument('--only_test', type=lambda x: (str(x).lower() == 'true'), default='false')
parser.add_argument('--model_path', type=str, default='.')

cfg_args = parser.parse_args()
