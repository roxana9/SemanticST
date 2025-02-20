#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:10:16 2024

@author: tower2
"""
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from .Semantic_layers import *
from .preprocess import preprocess_adj

# Define loss and optimizer
def L_dis(descriptors):
    """
    Computes the pairwise cosine similarity loss between descriptor vectors.

    This loss function encourages descriptors to be distinguishable by 
    penalising high similarity between different descriptor pairs.

    Args:
        descriptors (torch.Tensor): A tensor of shape (K, D) where K is the 
                                    number of descriptors, and D is the feature dimension.

    Returns:
        torch.Tensor: The total similarity loss across all descriptor pairs.

    
    """
    K = descriptors.shape[0]
    loss = 0
    
    for i in range(K-1):
        for j in range(i+1, K):
            numerator = torch.dot(descriptors[i], descriptors[j])
            denominator = torch.norm(descriptors[i]) * torch.norm(descriptors[j]) + 1e-8
            loss += numerator / denominator
    
    return loss

def L_dis_from_adjacency(descriptors, adjacency_matrix, Gk_graphs, lambda_factor=0.01):
    """
    Computes a combined loss based on descriptor similarity and adjacency constraints.

    This function calculates:
    - **Graph similarity loss**: Encourages distinguishability between descriptors.
    - **Node similarity loss**: Ensures that connected nodes in the graph remain similar.

    Args:
        descriptors (torch.Tensor): A tensor of shape (K, N, D), where K is the number of graphs,
                                    N is the number of nodes, and D is the feature dimension.
        adjacency_matrix (torch.Tensor): Binary adjacency matrix of shape (N, N) representing graph connectivity.
        Gk_graphs (list of torch.Tensor): A list of K graph representations, where each Gk_graph is of shape (N, D).
        lambda_factor (float, optional): Regularisation factor for node similarity loss (default: 0.01).

    Returns:
        torch.Tensor: The total loss, including both descriptor similarity and adjacency constraints.

   
    """
    K = descriptors.shape[0]
    num_nodes = descriptors.shape[1]
    loss = 0
    
    # Graph similarity loss
    for i in range(K-1):
        for j in range(i+1, K):
            numerator = torch.dot(descriptors[i], descriptors[j])
            denominator = torch.norm(descriptors[i]) * torch.norm(descriptors[j]) + 1e-8
            loss += numerator / denominator
    
    # Node similarity loss for neighbors
    for k in range(K):
        for i in range(num_nodes):
            neighbors = torch.nonzero(adjacency_matrix[i] == 1).squeeze()
            for neighbor in neighbors:
                if i != neighbor:  # Exclude self-loops
                    sim_neighbors = torch.norm(Gk_graphs[k][i] - Gk_graphs[k][neighbor])
                    loss += lambda_factor * sim_neighbors
    
    return loss

class SemanticGraphLearning:
    """
    A class for learning semantic graph representations from input data 
    using graph neural networks (GNNs).

    This class constructs a graph-based model, trains it using an adjacency 
    matrix and edge indices, and allows evaluation of new data.

    Attributes:
        data (numpy.ndarray or torch.Tensor): Input feature data.
        adj (torch.Tensor): Adjacency matrix representing graph structure.
        edge_index (torch.Tensor): Edge indices for sparse graph representation.
        device (str): Computing device ('cuda' or 'cpu').
        data_type (str): Type of data ('Slide' or 'Stereo' for direct edge usage, otherwise edges are derived).
        mini_batch (bool): Whether to use mini-batch training.
        hidden_dim1 (int): Dimension of the first hidden layer (default: 32).
        hidden_dim2 (int): Dimension of the second hidden layer (default: 32).
        input_dim_h (int): Intermediate input dimension (default: 32).
        out_dim (int): Output dimension of the model (default: 32).
        K (int): Number of layers in the graph model (default: 4).
        df (int): A hyperparameter for controlling feature dimensionality (default: 32).
        num_epochs (int): Number of training epochs (default: 1).
        features (torch.Tensor): Tensor containing input feature data.
        input_dim (int): Dimension of input features.
        A (torch.Tensor): Adjacency matrix used in training.
        model (GraphModel): Instantiated graph model for learning semantic representations.
        optimizer (torch.optim.Optimizer): Optimization algorithm (Adagrad).

    Methods:
        train():
            Trains the graph model, updates parameters using Adagrad, 
            and returns the learned semantic graphs.

        evaluate(data):
            Evaluates new input data using the trained model 
            and returns the corresponding semantic graphs.

   
    """
    def __init__(self, data, adj, edge_index, device,mini_batch=False, data_type="Slide"):
        self.data = data
        self.adj = adj
        self.edge_index = edge_index
        self.device = device
        self.data_type = data_type
        self.mini_batch=mini_batch
        self.hidden_dim1 = 32
        self.hidden_dim2 = 32
        self.input_dim_h = 32
        self.out_dim = 32
        self.K = 4
        self.df = 32  # example value
        self.num_epochs = 250
        self.features = torch.FloatTensor(data).to(device)
        N, f = self.features.shape
        self.input_dim = f
        self.A = adj
        
        if data_type in ['Stereo', 'Slide']:
            self.edge_index = edge_index
        else:
            self.edge_index, _ = dense_to_sparse(self.A)
        
        self.model = GraphModel(self.input_dim, self.input_dim_h, self.hidden_dim1, self.hidden_dim2, self.out_dim, self.K, self.df, self.edge_index, self.device)
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.001)
        
    def train(self):
        """
        Trains the graph model to learn semantic representations.

        Performs forward propagation, computes loss, backpropagates gradients, 
        and updates model parameters over the specified number of epochs.

        Returns:
            list: The final learned semantic graphs.
        """
        self.model.train()
        loss_values = []
        epoch_iter = range(self.num_epochs)
        if not self.mini_batch:
            epoch_iter = tqdm(epoch_iter, desc="Learning Semantic graphs", unit="epoch")
        for epoch in epoch_iter:
            self.optimizer.zero_grad()
            descriptors, Gk_graphs = self.model(self.features)
            loss = L_dis_from_adjacency(descriptors, self.A, Gk_graphs)
            loss.backward()
            self.optimizer.step()
            loss_values.append(loss.item())
            
            if epoch == self.num_epochs - 1:  # only save the last set of Gk graphs
                self.final_Gk_graphs = Gk_graphs
                self.semantic_graphs = self.final_Gk_graphs
        
        # print("Finish")
        # plt.figure(figsize=(10, 6))
        # plt.plot(loss_values)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Loss over Epochs')
        # plt.grid(True)
        # plt.show()
        
        return self.semantic_graphs
    
    def evaluate(self, data):
        """
        Evaluates the trained model on new data and generates semantic graphs.

        Args:
            data (numpy.ndarray or torch.Tensor): Input feature data.

        Returns:
            list: The generated semantic graphs from the trained model.
        """
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(data).to(self.device)
            descriptors, Gk_graphs = self.model(features)
            return Gk_graphs


