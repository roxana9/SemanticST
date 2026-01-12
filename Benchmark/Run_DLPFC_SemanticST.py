#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:08:58 2026

@author: tower4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:56:25 2024

@author: tower2
"""
from sklearn import metrics
import torch
import copy
import os
import random
import numpy as np
from semanticst.loading_batches import PrepareDataloader
from semanticst.loading_batches import Dataloader
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch.utils.data as data
from semanticst.main import Config
dtype = "Visium"  # Change to "h5ad" if necessary

from semanticst.SemanticST_main import Semantic as Trainer
import time
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

BASE_PATH = Path('/DATA1/Roxana/GraphST-main/Data/1.DLPFC')
#output_path = Path('/media/tower2/DATA4/Roxana/GraphST-main/domains_GraphST_DLPFC')
#sample_list = ['151673']

sample_list = ['151507', '151508', '151509', '151510', 
                '151669', '151670', '151671', '151672', 
                '151673', '151674', '151675', '151676']

plot_color=["#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]

start_time = time.time()


# Load R packages

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

numpy2ri.activate()
importr("mclust")

import numpy as np
import pandas as pd
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

ARI_by_seed = {}
NMI_by_seed = {}
HMO_by_seed = {}

seeds=[9]

# --- Run the function ---
#res_z0 = mclust_cluster_via_R_script(adata, "emb_pca", "mclust_decoder", G=7, model="EEE")
for s in seeds:
    ARI_list = [];NMI_list = [];HMO_list = []
    
    for sample_name in sample_list:
        print(sample_name)
        dir_input = Path(f'{BASE_PATH}/{sample_name}/')
        # dir_output = Path(f'{output_path}/')
        # dir_output.mkdir(parents=True, exist_ok=True)
        
        if sample_name in ['151669', '151670', '151671', '151672']:
            n_clusters = 5
        else:
            n_clusters = 7
        adata = sc.read_visium(f'{dir_input}')
        adata.var_names_make_unique()
        
        config=Config(device=device,dtype=dtype, use_mini_batch=False,seed=s)
        config_used = copy.copy(config)
        models = Trainer(adata,config)  
        adata=models.train()  # Train the model
    
        tool = 'mclust' # mclust, leiden, and louvain
        pca = PCA(n_components=20, random_state=int(s)) 
        embedding = pca.fit_transform(adata.obsm["emb_decoder"].copy())
        adata.obsm['emb_pca'] = embedding
    # clustering
        print("clustring")
        _ = mclust_cluster_via_R_script(
            adata,
            obsm_key='emb_pca',
            obs_key="domain",
            G=n_clusters,
            model="EEE",
            seed=s,
            subset_size=2000
        )        
        df_meta = pd.read_csv(f'{dir_input}/metadata.tsv', sep='\t')
        df_meta_layer = df_meta['layer_guess']
        adata.obs['ground_truth'] = df_meta_layer.values
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]
       # np.save(f"{dir_output}/domain_refined_{sample_name}.npy",adata.obs['domain'])
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
        sc.pl.embedding(adata, basis="spatial", color="domain",palette='magma' , show=False,legend_loc='none', title="SemanticST: ARI=%.4f"%ARI)
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
