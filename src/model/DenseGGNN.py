#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/3/3 17:16
# @Author   : Xiang Ling
# @Lab      : nesa.zju.edu.cn
# @File     : DenseGGNN.py
# ************************************

import torch
import torch.nn as nn

from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_geometric.utils import dense_to_sparse


class DenseGGNN(nn.Module):
    def __init__(self, out_channels, num_layers=1):
        super(DenseGGNN, self).__init__()
        self.model = GatedGraphConv(out_channels=out_channels, num_layers=num_layers)
    
    def forward(self, x, adj, **kwargs):
        B = x.size()[0]
        N = x.size()[1]
        D = x.size()[2]
        indices = []
        for i in range(B):
            edge_index = dense_to_sparse(adj[i])
            indices.append(edge_index[0] + i * N)
        edge_index = torch.cat(indices, dim=1)
        x = x.reshape(-1, D)
        output = self.model(x, edge_index)
        return output.reshape(B, N, -1)
