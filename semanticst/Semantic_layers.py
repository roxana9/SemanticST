#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:40:02 2024

@author: tower2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from tqdm import tqdm
from torchviz import make_dot
class GraphModel(nn.Module):
    def __init__(self, input_dim,input_dim_h, hidden_dim1, hidden_dim2, out_dim, K, df,edge_index, device):
        super(GraphModel, self).__init__()
        self.K = K
        self.df = df
        self.device = device
        
        # Define the transformation matrix Wh as a parameter to be learned
        self.Wh = nn.Parameter(torch.empty(input_dim, df, device=device))
        nn.init.xavier_uniform_(self.Wh)


        # Multiple MLPs for different semantic relations
        self.mlp_module = MLPLayers(2 * df, 2, K).to(device)
        
        # Autoencoder
        self.autoencoder = GraphAutoEncoder(input_dim_h, hidden_dim1, hidden_dim2, out_dim).to(device)
        self.edge_index=edge_index
    def forward(self, X):
        #descriptors = torch.empty(0, 32).to(device)
        descriptors=[]
        Gk_list = []
        h_0 = torch.mm(X, self.Wh)
        edge_list = self.edge_index
        edge_h = torch.cat((h_0[edge_list[0, :], :],h_0[edge_list[1, :], :]), dim=1)
       
        #num_edges = edge_index.shape[1]
        for k in range(self.K):
            Gk = self.mlp_module(edge_h, k)
            Gk2=Gk.squeeze(1)
            Gk_list.append(Gk2.detach() )
           # print(edge_index.shape)
           # print(h_0.shape)
            edge_weight = Gk.to(self.device)
            descriptor = self.autoencoder(h_0, self.edge_index, edge_weight=edge_weight)
            
            descriptors.append(descriptor.unsqueeze(0))
            
          #  descriptors = torch.cat((descriptors, descriptor.unsqueeze(0)), dim=0)
        descriptors = torch.cat(descriptors, dim=0) 
        return descriptors,Gk_list
class MLPLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLPLayers, self).__init__()
        self.mlps = nn.ModuleList()
        self.layer_usage_counter = [0] * num_layers
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.mlps.append(mlp)
            self._init_weights(mlp)
        
    def _init_weights(self, mlp):
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
    def forward(self, x, layer_idx):
        output = self.mlps[layer_idx](x) 
        return torch.sigmoid(output)

  
class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, out_dim=32):
        super(GraphAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder_layer1 = GCNConv(input_dim, hidden_dim1)
        self.encoder_layer2 = GCNConv(hidden_dim1, hidden_dim2)
        
        # Fully connected layer after global pooling
        self.fc = nn.Linear(hidden_dim2, out_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        # Encoder
        Z = F.relu(self.encoder_layer1(x, edge_index, edge_weight))
        Z = F.relu(self.encoder_layer2(Z, edge_index, edge_weight))
        
        # Fully connected layer after global pooling
        pooled_Z = torch.mean(Z, dim=0)  # global average pooling

        out = self.fc(pooled_Z)

        return out

