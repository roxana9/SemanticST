#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:55:53 2024

@author: tower2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:30:46 2024

@author: tower2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:56:25 2024

@author: tower2
"""
import os 
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn import metrics
import time
import SEDR
from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from sklearn.decomposition import PCA

numpy2ri.activate()

def mclust_cluster_via_R_script(adata, obsm_key, obs_key, G=7, model="EEE", seed=2025, subset_size=2000):
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, d = X.shape
    if not np.isfinite(X).all():
        raise ValueError(f"{obsm_key} contains NaN/Inf")

    ro.globalenv["X_py"] = X
    ss = min(int(subset_size), n)
    ro.globalenv["SS"] = ss

    r_code = f"""
    suppressPackageStartupMessages(library(mclust))
    set.seed({int(seed)})

    # Build matrix in R from X_py (already numeric)
    Xr <- matrix(X_py, nrow=nrow(X_py), ncol=ncol(X_py))
    colnames(Xr) <- paste0("V", seq_len(ncol(Xr)))

    # sanity prints
    cat("dim(Xr) =", dim(Xr), "\\n")
    cat("ncol(Xr) =", ncol(Xr), " length(colnames) =", length(colnames(Xr)), "\\n")

    init_list <- list(subset = sample.int(nrow(Xr), SS))

    # Run Mclust (note: modelNames is a plain string)
    res <- Mclust(Xr, G={int(G)}, modelNames="{model}", initialization=init_list)

    res
    """

    # Execute and get the returned object
    res = ro.r(r_code)

    labels = np.array(res.rx2("classification"), dtype=int) - 1
    adata.obs[obs_key] = pd.Categorical(labels.astype(str))

    return res



BASE_PATH = Path('/DATA1/Roxana/GraphST-main/Data/1.DLPFC')
#output_path = Path('/media/tower2/DATA4/Roxana/SEDR-master/Results_DLPFC2')

sample_list = ['151507', '151508', '151509', '151510', 
                '151669', '151670', '151671', '151672', 
                '151673', '151674', '151675', '151676']

plot_color=["#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
start_time = time.time()
ARI_by_seed = {}
NMI_by_seed = {}
HMO_by_seed = {}

seeds=[9]
for s in seeds:
    ARI_list = [];NMI_list = [];HMO_list = []
    
    SEDR.fix_seed(int(s))
    for sample_name in sample_list:
        print(sample_name)
        dir_input = Path(f'{BASE_PATH}/{sample_name}/')
        # dir_output = Path(f'{output_path}/{sample_name}/')
        # dir_output.mkdir(parents=True, exist_ok=True)
        df_meta = pd.read_csv(f'{dir_input}/metadata.tsv', sep='\t')
        df_meta_layer = df_meta['layer_guess']
        if sample_name in ['151669', '151670', '151671', '151672']:
            n_clusters = 5
        else:
            n_clusters = 7
        adata = sc.read_visium(f'{dir_input}')
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = df_meta_layer.values
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
        adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)
        
        adata_X = PCA(n_components=200, random_state=s).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        graph_dict = SEDR.graph_construction(adata, 12)
        print(graph_dict)
        sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
        using_dec = True
        if using_dec:
            sedr_net.train_with_dec(N=1)
        else:
            sedr_net.train_without_dec(N=1)
        sedr_feat, _, _, _ = sedr_net.process()
        adata.obsm['SEDR'] = sedr_feat
        _ = mclust_cluster_via_R_script(
            adata,
            obsm_key='SEDR',
            obs_key="domain",
            G=n_clusters,
            model="EEE",
            seed=s,
            subset_size=2000
        )        
      
        
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]
        ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
        NMI = metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
        HMO= metrics.homogeneity_score(adata.obs['domain'], adata.obs['ground_truth'])
        print('===== Project: {} ARI score: {:.3f}'.format(sample_name, ARI))
        ARI_list.append(ARI)
        print('===== Project: {} NMI score: {:.3f}'.format(sample_name, NMI))
        NMI_list.append(NMI)
        print('===== Project: {} HMO score: {:.3f}'.format(sample_name, HMO))
        HMO_list.append(HMO)
        plt.rcParams["figure.figsize"] = (3,3)
        sc.pl.embedding(adata, basis="spatial", color="domain",palette='magma' , show=False,legend_loc='none', title="SEDR: ARI=%.4f"%ARI)
        plt.gca().invert_yaxis()  # This will invert the y-axis
        plt.axis('off')
        #plt.savefig(f"{dir_output}/pred.png", dpi=300)
        plt.show()
        end_time = time.time()   
        total_running_time = end_time - start_time  # Calculate total running tim    
        print(f"Total running time for processing all datasets: {total_running_time:.2f} seconds.")  
    ARI_by_seed[int(s)] = ARI_list
    NMI_by_seed[int(s)] = NMI_list
    HMO_by_seed[int(s)] = HMO_list
