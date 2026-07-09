#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:31:26 2024

@author: Roxana
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .preprocess import (
    preprocess,
    construct_interaction,
    construct_interaction_KNN,
    construct_interaction_KNN_edge_index,
    construct_sparse_graph,
    sparse_mx_to_torch_sparse_tensor,
    fix_seed,
)
from .model import Encoder, DeepMinCutModel
from .Semantic_graphs import SemanticGraphLearning


class Semantic_batches():
    """
    A class for learning semantic representations from spatial transcriptomics (ST) data
    using graph-based methods.

    This class constructs spatial graphs, applies deep learning-based feature extraction,
    and integrates reconstruction loss and MinCut loss to enhance the learned embeddings.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spatial transcriptomics data.

    train_loader : torch.utils.data.DataLoader
        DataLoader for training batches of spatial transcriptomics data.

    test_loader : torch.utils.data.DataLoader
        DataLoader for testing batches of spatial transcriptomics data.

    num_iter : int
        Number of iterations (batches) per epoch.

    config : object
        Configuration object containing device, data type, and other hyperparameters.

    learning_rate : float, optional, default=0.001
        Learning rate for optimizing the model.

    weight_decay : float, optional, default=0.00
        Regularization parameter controlling weight decay in the optimizer.

    epochs : int, optional, default=1
        Number of training epochs.

    dim_output : int, optional, default=64
        Dimensionality of the output embeddings.

    alpha : float, optional, default=1
        Weight factor controlling the influence of the reconstruction loss in representation learning.

    beta : float, optional, default=0.1
        Weight factor controlling the influence of the **MinCut loss** in representation learning.

    Integration : bool, optional, default=True
        Whether to integrate additional data sources in representation learning.

    Attributes
    ----------
    dim_input : int
        Dimensionality of the input features.

    device : str
        Computation device (`'cpu'` or `'cuda'`), retrieved from config.

    adj : torch.Tensor
        Adjacency matrix representing the spatial interactions between spots.

    edge_index : torch.Tensor
        Edge list representing graph connectivity.

    G : list
        List of semantic graphs constructed during training.

    emb_rec : np.ndarray
        Learned representations from the decoder.

    emb_h : np.ndarray
        Learned representations from the encoder.

    Methods
    -------
    train():
        Trains the semantic graph learning model on ST data using reconstruction loss and MinCut loss.

    test():
        Evaluates the trained model and extracts representations.

    Returns
    -------
    anndata.AnnData
        Updated AnnData object with learned embeddings stored in `adata.obsm['emb_decoder']`
        and `adata.obsm['emb_encoder']`.
    """

    def __init__(
        self,
        adata,
        train_loader,
        test_loader,
        num_iter,
        config,
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=1000,
        dim_output=64,
        alpha=1,
        beta=0.1,
        Integration=True
    ):
        self.adata = adata
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.datatype = self.config.dtype
        self.alpha = alpha
        self.beta = beta
        self.Integration = Integration
        self.device = self.config.device
        fix_seed(self.config.seed)
        batch, _ids = next(iter(train_loader))
        self.num_iter = num_iter
        self.dim_input = batch[:, :-2].shape[1]
        self.dim_output = dim_output
        print("\n🚀 Welcome to SemanticST! 🚀\n")
        print("📢 Recommendation: If your dataset contains more than 40000 spots or cells, we suggest using **mini-batch training** for efficiency.")

    def train(self):
        print("\n✅ Using Mini-Batch Training for better efficiency! 🏋️‍♂️")
        self.min_model = DeepMinCutModel()

        print('Begin to train ST data...')
        self.loss = 0
        total_loss = 0
        batch_idx = 0
        self.model = Encoder(self.dim_input, self.dim_output, num_graphs=4).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        for batch_idx, (data, ids) in enumerate(tqdm(self.train_loader, total=self.num_iter, desc="Training Progress")):
            spot_data = data.float()
            num_nodes = data.shape[0]
            coor = spot_data[:, -2:]
            coor = coor.to(torch.int)
            W = []
            self_loops = torch.arange(num_nodes, device=self.device)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            if self.datatype in ['Stereo', 'Slide']:
                self.adj = construct_sparse_graph(coor)                    # scipy sparse, stays sparse
                gn = self.adj + sp.eye(self.adj.shape[0])
                self.graph_neigh = sparse_mx_to_torch_sparse_tensor(gn).to(self.device)   # binary mask
                self.edge_index = construct_interaction_KNN_edge_index(coor).to(self.device)
            else:
                self.adj, self.graph_neigh = construct_interaction(coor)
                self.graph_neigh = torch.FloatTensor(self.graph_neigh + np.eye(self.adj.shape[0])).to(self.device)
                self.adj = torch.FloatTensor(self.adj).to(self.device)
                self.edge_index = self.adj.nonzero().t()
                self.edge_index = torch.cat((self.edge_index, self_loops), dim=1)

            self.loop_weight = torch.full((num_nodes,), 0.1).to(self.device)
            self.sg_learning = SemanticGraphLearning(spot_data[:, :-2], self.adj, self.edge_index, self.device, self.config.use_mini_batch, self.datatype)
            self.G = self.sg_learning.train()

            for i in range(len(self.G)):
                if self.datatype in ['Stereo', 'Slide']:
                    self.new_edge_weight = self.G[i]
                else:
                    self.new_edge_weight = torch.cat((self.G[i], self.loop_weight))
                W.append(self.new_edge_weight)
            spot_data = torch.FloatTensor(spot_data[:, :-2])
            feature = spot_data.to(self.device)
            self.adj_weighted = self.edge_weights_to_sparse(num_nodes)

            for epoch in range(self.epochs):
                self.model.train()
                self.hiden_feat, self.emb, self.g, self.hidden_embeddings, self.attn_weights = self.model(feature, self.edge_index, self.graph_neigh, W)
                self.loss_feat = F.mse_loss(feature, self.emb)
                self.loss_deep_mincut = self.min_model.deep_mincut_loss(self.hiden_feat, self.adj_weighted)
                loss = self.alpha * self.loss_feat + self.beta * self.loss_deep_mincut
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.data.item()

            batch_idx += 1
            self.loss = total_loss / (batch_idx + 1)

            del self.edge_index, self.adj, spot_data, self.graph_neigh, self.G
            del self.loop_weight, self.adj_weighted
        print("Optimization finished for ST data!")

        with torch.no_grad():
            self.model.eval()
            Emb_e = []
            Emb_d = []
            for batch_idx, (data, ids) in enumerate(tqdm(self.test_loader, desc="Testing Progress", unit="batch")):
                spot_data = data.float()
                num_nodes = data.shape[0]
                coor = spot_data[:, -2:]
                W = []
                self_loops = torch.arange(num_nodes, device=self.device)
                self_loops = torch.stack([self_loops, self_loops], dim=0)
                if self.datatype in ['Stereo', 'Slide']:
                    self.adj, self.graph_neigh = construct_interaction_KNN(coor)
                    self.graph_neigh = torch.FloatTensor(self.graph_neigh + np.eye(self.adj.shape[0])).to(self.device)
                    self.edge_index = construct_interaction_KNN_edge_index(coor).to(self.device)
                    self.adj = torch.FloatTensor(self.adj).to(self.device)
                else:
                    self.adj, self.graph_neigh = construct_interaction(coor)
                    self.graph_neigh = torch.FloatTensor(self.graph_neigh + np.eye(self.adj.shape[0])).to(self.device)
                    self.adj = torch.FloatTensor(self.adj).to(self.device)
                    self.edge_index = self.adj.nonzero().t()
                    self.edge_index = torch.cat((self.edge_index, self_loops), dim=1)

                self.loop_weight = torch.full((num_nodes,), 0.1).to(self.device)
                self.sg_learning = SemanticGraphLearning(spot_data[:, :-2], self.adj, self.edge_index, self.device, self.config.use_mini_batch, self.datatype)
                self.G = self.sg_learning.evaluate(spot_data[:, :-2])
                for i in range(len(self.G)):
                    if self.datatype in ['Stereo', 'Slide']:
                        self.new_edge_weight = self.G[i]
                    else:
                        self.new_edge_weight = torch.cat((self.G[i], self.loop_weight))
                    W.append(self.new_edge_weight)
                spot_data = torch.FloatTensor(spot_data[:, :-2])
                feature = spot_data.to(self.device)
                self.emb_rec = self.model(feature, self.edge_index, self.graph_neigh, W)[1].detach()
                self.emb_h = self.model(feature, self.edge_index, self.graph_neigh, W)[0].detach()
                Emb_d.append(self.emb_rec.cpu().numpy())
                Emb_e.append(self.emb_h.cpu().numpy())
            Emb_d = np.concatenate(Emb_d, axis=0)
            Emb_e = np.concatenate(Emb_e, axis=0)
            self.adata.obsm['emb_decoder'] = Emb_d
            self.adata.obsm['emb_encoder'] = Emb_e

        return self.adata

    def edge_weights_to_sparse(self, num_nodes):
        A = torch.sparse_coo_tensor(
            self.edge_index, self.new_edge_weight,
            (num_nodes, num_nodes), device=self.device,
        )
        return (A + A.t()).coalesce()


class Semantic():
    """
    A class for training a semantic graph learning model on full spatial transcriptomics (ST) data
    without mini-batch training.

    This class constructs spatial graphs, applies deep learning-based feature extraction,
    and integrates reconstruction loss and MinCut loss to enhance the learned embeddings.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spatial transcriptomics data.

    config : object
        Configuration object containing device, data type, and other hyperparameters.

    learning_rate : float, optional, default=0.001
        Learning rate for optimizing the model.

    weight_decay : float, optional, default=0.00
        Regularization parameter controlling weight decay in the optimizer.

    epochs : int, optional, default=1000
        Number of training epochs.

    dim_input : int, optional, default=3000
        Dimensionality of the input features.

    dim_output : int, optional, default=64
        Dimensionality of the output embeddings.

    alpha : float, optional, default=10
        Weight factor controlling the influence of the reconstruction loss in representation learning.

    beta : float, optional, default=0.1
        Weight factor controlling the influence of the **MinCut loss** in representation learning.

    Integration : bool, optional, default=True
        Whether to integrate additional data sources in representation learning.

    Attributes
    ----------
    device : str
        Computation device (`'cpu'` or `'cuda'`), retrieved from config.

    adj : torch.Tensor
        Adjacency matrix representing the spatial interactions between spots.

    edge_index : torch.Tensor
        Edge list representing graph connectivity.

    G : list
        List of semantic graphs constructed during training.

    emb_rec : np.ndarray
        Learned representations from the decoder.

    emb_h : np.ndarray
        Learned representations from the encoder.

    Methods
    -------
    train():
        Trains the semantic graph learning model on the full dataset without mini-batching.

    Returns
    -------
    anndata.AnnData
        Updated AnnData object with learned embeddings stored in `adata.obsm['emb_decoder']`
        and `adata.obsm['emb_encoder']`.
    """

    def __init__(self,
        adata, config,
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=1000,
        dim_input=3000,
        dim_output=64,
        alpha=10,
        beta=0.1,
        lambda_factor=0.1,
        Integration=True
    ):
        self.adata = adata.copy()
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = self.config.seed
        self.alpha = alpha
        self.beta = beta
        self.Integration = Integration
        self.datatype = self.config.dtype
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.device = self.config.device
        self.lambda_factor = lambda_factor
        fix_seed(self.config.seed)
        print("\n🚀 Welcome to SemanticST! 🚀\n")
        print("📢 Recommendation: If your dataset contains more than 40000 spots or cells, we suggest using **mini-batch training** for efficiency.")

    def train(self):
        print("\n✅ Using Full Dataset Training (No Mini-Batching). 🔥")
        preprocess(self.adata, self.datatype)
        self.min_model = DeepMinCutModel()
        print('Begin to train ST data...')
        data = self.adata.obsm['feat'].copy()
        self.dim_input = data.shape[1]
        num_nodes = data.shape[0]
        coor = self.adata.obsm['spatial']
        coor = torch.from_numpy(coor).to(device=self.device, dtype=torch.int)
        W = []
        self_loops = torch.arange(num_nodes, device=self.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        if self.datatype in ['Stereo', 'Slide']:
            self.adj = construct_sparse_graph(coor)                    # scipy sparse, stays sparse
            gn = self.adj + sp.eye(self.adj.shape[0])
            self.graph_neigh = sparse_mx_to_torch_sparse_tensor(gn).to(self.device)   # binary mask
            self.edge_index = construct_interaction_KNN_edge_index(coor).to(self.device)
        else:
            self.adj, self.graph_neigh = construct_interaction(coor)
            self.graph_neigh = torch.FloatTensor(self.graph_neigh + np.eye(self.adj.shape[0])).to(self.device)
            self.adj = torch.FloatTensor(self.adj).to(self.device)
            self.edge_index = self.adj.nonzero().t()
            self.edge_index = torch.cat((self.edge_index, self_loops), dim=1)

        self.loop_weight = torch.full((num_nodes,), 0.1).to(self.device)
        self.sg_learning = SemanticGraphLearning(data, self.adj, self.edge_index, self.device, self.config.use_mini_batch, self.datatype, lambda_factor=self.lambda_factor)
        self.sg_learning.train()
        self.G = self.sg_learning.evaluate(data)

        print("Semantic Graph Learning Completed")
        num_graphs = len(self.G[1])

        self.model = Encoder(self.dim_input, self.dim_output, num_graphs=num_graphs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        for i in range(len(self.G)):
            if self.datatype in ['Stereo', 'Slide']:
                self.new_edge_weight = self.G[i]
            else:
                self.new_edge_weight = torch.cat((self.G[i], self.loop_weight))
            W.append(self.new_edge_weight)
        data = torch.FloatTensor(data)
        feature = data.to(self.device)
        self.adj_weighted = self.edge_weights_to_sparse(num_nodes)

        for epoch in tqdm(range(self.epochs), desc="Feature Learning Epochs"):
            self.model.train()
            self.hiden_feat, self.emb, self.g, self.hidden_embeddings, self.attn_weights = self.model(feature, self.edge_index, self.graph_neigh, W)
            self.loss_feat = F.mse_loss(feature, self.emb)
            self.loss_deep_mincut = self.min_model.deep_mincut_loss(self.hiden_feat, self.adj_weighted)
            loss = self.alpha * self.loss_feat + self.beta * self.loss_deep_mincut
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            data = torch.FloatTensor(data)
            feature = data.to(self.device)
            self.emb_rec = self.model(feature, self.edge_index, self.graph_neigh, W)[1].detach()
            self.emb_h = self.model(feature, self.edge_index, self.graph_neigh, W)[0].detach()
            self.adata.obsm['emb_decoder'] = self.emb_rec.cpu().numpy()
            self.adata.obsm['emb_encoder'] = self.emb_h.cpu().numpy()
        return self.adata

    def edge_weights_to_sparse(self, num_nodes):
        A = torch.sparse_coo_tensor(
            self.edge_index, self.new_edge_weight,
            (num_nodes, num_nodes), device=self.device,
        )
        return (A + A.t()).coalesce()

    def cosine_similarity(self, pred_sp, emb_sp):   # pred_sp: spot x gene; emb_sp: spot x gene
        """
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.
        """
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        if torch.any(torch.isnan(M)):
            M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)
        return M