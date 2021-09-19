#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:42:16 2021

@author: Gong Dongsheng
"""

import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.mm(self.dropout(inputs), self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_layers, n_features, hidden_dim, dropout, n_classes):
        super(GCN, self).__init__()
        if n_layers == 1:
            self.first_layer = GraphConvolution(n_features, n_classes, dropout)
        else:
            self.first_layer = GraphConvolution(n_features, hidden_dim, dropout)
            self.last_layer = GraphConvolution(hidden_dim, n_classes, dropout)
            if n_layers > 2:
                self.gc_layers = nn.ModuleList([
                    GraphConvolution(hidden_dim, hidden_dim, 0) for _ in range(n_layers - 2)
                ])
            
        self.n_layers = n_layers
        self.relu = nn.ReLU()
    
    def forward(self, inputs, adj):
        if self.n_layers == 1:
            x = self.first_layer(inputs, adj)
        else:
            x = self.relu(self.first_layer(inputs, adj))
            if self.n_layers > 2:
                for i, layer in enumerate(self.gc_layers):
                    x = self.relu(layer(x, adj))
            x = self.last_layer(x, adj)
        return x
