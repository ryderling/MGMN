#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/7/13 22:34
# @Author   : Xiang Ling
# @Lab      : nesa.zju.edu.cn
# @File     : ged_config.py
# ************************************
import argparse

parser = argparse.ArgumentParser(description="Multi-Level Graph Matching Network for the Graph-Graph Regression task")

parser.add_argument('--data_dir', type=str, default='../data/GED/', help='root directory for the data')
parser.add_argument('--dataset', type=str, default="aids700nef", help='indicate the specific data set (aids700nef/linux)')

parser.add_argument("--task", type=str, default='regression', help="classification/regression")

parser.add_argument("--filters", type=str, default='100_100_100', help="filters (neurons) for graph neural networks")
parser.add_argument("--conv", type=str, default='gcn', help="one kind of graph neural networks")
parser.add_argument("--match", type=str, default='node-graph', help="indicating the match method")
parser.add_argument("--perspectives", type=int, default=100, help='number of perspectives for node-graph matching')
parser.add_argument("--match_agg", type=str, default='bilstm', help="aggregator")
parser.add_argument("--hidden_size", type=int, default=100, help='hidden size for the graph-level embedding')

# global-level information
parser.add_argument("--global_flag", type=lambda x: (str(x).lower() == 'true'), default='True', help="Whether use global info ")
parser.add_argument("--global_agg", type=str, default='fc_max_pool', help="aggregation function for global level gcn ")

# training parameters for classification tasks
parser.add_argument('--iterations', type=int, default=10000, help='number of training epochs')
parser.add_argument('--iter_val_start', type=int, default=9000)
parser.add_argument('--iter_val_every', type=int, default=10)
parser.add_argument('--inclusive', type=lambda x: (str(x).lower() == 'true'), default='True', help='True')

parser.add_argument("--batch_size", type=int, default=128, help="Number of graph pairs per batch.")
parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")

# others
parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
parser.add_argument('--log_path', type=str, default='../GEDLogs/', help='path for log file')
parser.add_argument('--repeat_run', type=int, default=1, help='indicated the index of repeat run')

ged_args = parser.parse_args()
