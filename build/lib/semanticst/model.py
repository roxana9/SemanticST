import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
class AttentionMechanism(nn.Module):
    """
   Implements an attention mechanism to compute attention scores over input embeddings.

   Args:
       input_dim (int): Dimension of input features.
       hidden_dim (int): Hidden dimension used in the attention mechanism.

   Methods:
       forward(z): Computes attention scores over input embeddings.
   """
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim, bias=True)
        self.q = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, z):
        """
        Computes attention scores over input embeddings.

        Args:
            z (torch.Tensor): Input tensor of shape (N, input_dim).

        Returns:
            torch.Tensor: Attention scores of shape (N, 1).
        """
        h = torch.tanh(self.W(z))  # Non-linear transformation
        attn_weights = self.q(h)  # Compute attention scores
        return attn_weights

class GraphEmbeddingFusion(nn.Module):
    """
    Fuses multiple semantic graph embeddings using an attention mechanism.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden dimension used in attention computation.
        num_graphs (int): Number of semantic graphs.

    Methods:
        forward(*z_list): Computes fused embedding and attention weights.
    """

    def __init__(self, input_dim, hidden_dim, num_graphs):
        super(GraphEmbeddingFusion, self).__init__()
        # Create an attention mechanism for each semantic graph using ModuleList
        self.attention_modules = nn.ModuleList([AttentionMechanism(input_dim, hidden_dim) for _ in range(num_graphs)])

    def forward(self, *z_list):
        """
        Fuses multiple graph embeddings using learned attention weights.

        Args:
            *z_list (torch.Tensor): List of semantic graph embeddings.

        Returns:
            tuple: (Fused embedding, Attention weights)
        """
        # Compute attention weights for each embedding
        attn_scores = [self.attention_modules[i](z) for i, z in enumerate(z_list)]
        attn_scores = torch.cat(attn_scores, dim=1)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Fuse embeddings using the attention weights
        fused_embedding = sum(attn_weights[:, i].unsqueeze(1) * z_list[i] for i in range(len(z_list)))
        return fused_embedding, attn_weights
class DeepMinCutModel(nn.Module):
    """
   Implements the Deep MinCut clustering model using Gumbel-Softmax sampling.

   Methods:
       sample_gumbel_softmax(embeddings, temperature): Samples a probability distribution using Gumbel-Softmax.
       deep_mincut_loss(embeddings, adjacency, temperature): Computes the MinCut loss for community detection.
   """
    def __init__(self,):
        super(DeepMinCutModel, self).__init__()

    def sample_gumbel_softmax(self, embeddings, temperature=1):
        """
        Samples a probability distribution using Gumbel-Softmax.

        Args:
            embeddings (torch.Tensor): Input embeddings.
            temperature (float): Temperature parameter for softmax.

        Returns:
            torch.Tensor: Sampled probability distribution.
        """
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(embeddings)))
        logits = (embeddings + gumbel_noise) / temperature
        P = torch.softmax(logits, dim=1)
        return P

    def deep_mincut_loss(self, embeddings, adjacency, temperature=1):
        """
        Computes the Deep MinCut loss for graph-based clustering.

        Args:
            embeddings (torch.Tensor): Input embeddings.
            adjacency (torch.Tensor): Adjacency matrix of the graph.
            temperature (float): Softmax temperature parameter.

        Returns:
            torch.Tensor: Computed MinCut loss.
        """
        P = self.sample_gumbel_softmax(embeddings, temperature)
        C = torch.matmul(P.T, torch.matmul(adjacency, P))
        d = torch.diag(C)
        q = torch.sum(C, dim=1)
        l = (q - d) / q
        normcut_loss = torch.sum(l)
        return torch.log(normcut_loss)
    
class AvgReadout(nn.Module):
    """
    Computes the average readout (global representation) of node embeddings.

    Methods:
        forward(emb, mask): Computes global embedding from node embeddings.
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        """
        Computes global embedding from node embeddings.

        Args:
            emb (torch.Tensor): Node embeddings of shape (N, D).
            mask (torch.Tensor, optional): Masking matrix.

        Returns:
            torch.Tensor: Normalised global embedding.
        """
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1)

class Encoder(Module):
    """
    Implements a graph convolutional neural network (GCN) encoder with attention-based graph fusion.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        num_graphs (int): Number of semantic graphs.
        dropout (float): Dropout rate.
        act (function): Activation function.

    Methods:
        forward(x, edge_index, graph_neigh, all_edge_weights): Encodes node embeddings and fuses them.
    """
    def __init__(self, in_features, out_features,num_graphs, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.num_graphs = num_graphs  # Number of semantic graphs
        self.conv1 = GCNConv(in_features, out_features)
        self.conv2 = GCNConv(out_features, in_features)
        self.read = AvgReadout()
        self.fusion_model = GraphEmbeddingFusion(self.out_features, 64,num_graphs)

    def forward(self, x, edge_index,graph_neigh, all_edge_weights):
        """
        Encodes node embeddings using GCNs and fuses multiple semantic graph embeddings.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph edge indices.
            graph_neigh (torch.Tensor): Graph neighborhood structure.
            all_edge_weights (list of torch.Tensor): List of edge weight tensors for different graphs.

        Returns:
            tuple: (Fused hidden embedding, final embedding, global embedding, list of hidden embeddings, attention weights)
        """
        hidden_embeddings = []
        for edge_weight in all_edge_weights:
            # First GCN layer
            x1 = F.dropout(x, self.dropout, training=self.training)
            x1 = self.conv1(x1, edge_index, edge_weight)
            x1 = self.act(x1)

            hidden_emb = x1  # Save the output of the first layer as hidden embeddings

          
            hidden_embeddings.append(hidden_emb)
        
        # Fuse the hidden embeddings
        fused_hidden_embedding,attn_weights  = self.fusion_model(*hidden_embeddings)
        # Decoder
        x1 = F.dropout(fused_hidden_embedding, self.dropout, training=self.training)
        h = self.conv2(x1, edge_index)
        x1 = self.act(h)
        g = self.read(fused_hidden_embedding, graph_neigh)
        emb=x1
        
        return fused_hidden_embedding, emb, g,hidden_embeddings,attn_weights 
    

          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    