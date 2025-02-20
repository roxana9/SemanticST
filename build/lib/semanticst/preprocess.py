import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 

def construct_interaction(coor, n_neighbors=5):
    """Constructing spot-to-spot interactive graph"""
    if isinstance(coor, torch.Tensor):
        coor = coor.cpu().numpy()
    position = coor
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    if np.any(np.isnan(distance_matrix)):
        print("Warning: The distance matrix contains NaN values.")

    if np.any(distance_matrix < 0):
        print("Warning: The distance matrix contains negative values.")
    n_spot = distance_matrix.shape[0]
    
   # adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
       
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    graph_neigh= interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
   # print('Graph constructed!')

  #  adata.obsm['adj'] = adj
    return adj,graph_neigh
    
def construct_interaction_KNN(coor, n_neighbors=5):
    if isinstance(coor, torch.Tensor):
        coor = coor.cpu().numpy()
    position = coor
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    graph_neigh = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
   # adata.obsm['adj'] = adj
    #print('Graph constructed!')
    
    return adj,graph_neigh
def construct_interaction_KNN_edge_index(coor, n_neighbors=5):
    if isinstance(coor, torch.Tensor):
        coor = coor.cpu().numpy()
    position = coor
   
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)
    _, indices = nbrs.kneighbors(position)
    
    # Prepare edge_index data
    # Include self-references from the first column for self-loops
    rows = np.repeat(np.arange(indices.shape[0]), n_neighbors + 1)
    cols = indices.flatten()  # Include self-references for self-loops

    # Add edges in both directions to make it undirected
    undirected_rows = np.hstack([rows, cols])
    undirected_cols = np.hstack([cols, rows])

    # Remove duplicates
    edge_pairs = np.vstack([undirected_rows, undirected_cols])
    unique_edge_pairs = np.unique(edge_pairs, axis=1)
    
 
    # Convert to PyTorch tensor
    edge_index_tensor = torch.LongTensor(unique_edge_pairs)
    #print("Done!")
    return edge_index_tensor
def preprocess(adata,dtype):
    
    if dtype!='Xenium':
        
        # sc.pp.filter_cells(adata, min_genes=20)
        # sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata_Vars =  adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ] 
        adata.obsm['feat'] = feat
  
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ] 
    adata.obsm['feat'] = feat
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    
    
