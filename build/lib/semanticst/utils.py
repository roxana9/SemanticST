import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=12345):
    
    """
    Performs clustering using the Mclust algorithm in R.

    This function applies Gaussian mixture model-based clustering via `mclust` from R
    on the given AnnData object's embeddings.

    Args:
        adata (AnnData): The annotated data matrix.
        num_cluster (int): The number of clusters.
        modelNames (str, optional): Model specification for `mclust`. Default is 'EEE'.
        used_obsm (str, optional): Key in `adata.obsm` to use for clustering. Default is 'emb_pca'.
        random_seed (int, optional): Random seed for reproducibility. Default is 12345.

    Returns:
        AnnData: Updated `adata` object with Mclust clustering results stored in `adata.obs['mclust']`.

    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, seed,n_clusters, radius=50, key='emb', method='mclust'):
    
    """
    Applies dimensionality reduction (PCA) and clustering using Mclust.

    This function first reduces the feature space using PCA and then clusters
    the data using the Mclust algorithm in R.

    Args:
        adata (AnnData): The annotated data matrix.
        seed (int): Random seed for reproducibility.
        n_clusters (int): The number of clusters.
        radius (int, optional): Unused in current implementation but can be useful for spatial clustering. Default is 50.
        key (str, optional): Key in `adata.obsm` to use as input embeddings. Default is 'emb'.
        method (str, optional): Clustering method. Currently, only 'mclust' is supported. Default is 'mclust'.

    Returns:
        None: Updates `adata.obs` in place with clustering assignments under `adata.obs['domain']`.
    """
    pca = PCA(n_components=20, random_state=42) 
    embedding = pca.fit_transform(adata.obsm[key].copy())
    #adata.obsm['emb_pca']=adata.obsm[key].copy()
    adata.obsm['emb_pca'] = embedding
    
    
    adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters,random_seed=seed)
    adata.obs['domain'] = adata.obs['mclust']
   
       