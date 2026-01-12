import os
import matplotlib.pyplot as plt
import scanpy as sc
import deepstkit as dt
import pandas as pd
from sklearn import metrics
# ========== Configuration ==========
SEED = 9                    # Random seed for reproducibility
DATA_DIR = "/DATA1/Roxana/GraphST-main/Data/1.DLPFC"   # Directory containing spatial data
SAMPLE_ID = "151673"         # Sample identifier to analyze
RESULTS_DIR = "../Results"   # Directory to save outputs
N_DOMAINS = 7                # Expected number of spatial domains
ARI_list=[]
# ========== Initialize Analysis ==========
# Set random seed and initialize DeepST
dt.utils_func.seed_torch(seed=SEED)
sample_list = ['151507', '151508', '151509', '151510', 
                '151669', '151670', '151671', '151672', 
                '151673', '151674', '151675', '151676']

for sample_name in sample_list:
    print(sample_name)
# Create DeepST instance with analysis parameters
    deepst = dt.main.run(
        save_path=RESULTS_DIR,
        task="Identify_Domain",  # Spatial domain identification
        pre_epochs=500,          # Pretraining iterations
        epochs=500,              # Main training iterations
        use_gpu=True             # Accelerate with GPU if available
    )
    
    # ========== Data Loading & Preprocessing ==========
    # (Optional) Load spatial transcriptomics data (Visium platform)
    # e.g. adata = anndata.read_h5ad("*.h5ad"), this data including .obsm['spatial']
    adata = deepst._get_adata(
        platform="Visium",
        data_path=DATA_DIR,
        data_name=sample_name
    )
    
    # Optional: Incorporate H&E image features (skip if not available)
    # adata = deepst._get_image_crop(adata, data_name=SAMPLE_ID)
    
    # ========== Feature Engineering ==========
    # Data augmentation (skip morphological if no H&E)
    adata = deepst._get_augment(
        adata,
        spatial_type="BallTree",
        use_morphological = False  # Set True if using H&E features
    )
    
    # Construct spatial neighborhood graph
    graph_dict = deepst._get_graph(
        adata.obsm["spatial"],
        distType="KDTree"        # Spatial relationship modeling
    )
    
    # Dimensionality reduction
    data = deepst._data_process(
        adata,
        pca_n_comps=200          # Reduce to 200 principal components
    )
    
    # ========== Model Training ==========
    # Train DeepST model and obtain embeddings
    deepst_embed = deepst._fit(
        data=data,
        graph_dict=graph_dict
    )
    adata.obsm["DeepST_embed"] = deepst_embed
    if sample_name in ['151669', '151670', '151671', '151672']:
        n_clusters = 5
    else:
        n_clusters = 7
    
    # ========== Spatial Domain Detection ==========
    # Cluster spots into spatial domains
    adata = deepst._get_cluster_data(
        adata,
        n_domains=n_clusters,     # Expected number of domains
        priori=True              # Use prior knowledge if available
    )
    df_meta = pd.read_csv(f'{DATA_DIR}/{sample_name}/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    # ========== Visualization & Output ==========
    # Plot spatial domains
    sc.pl.spatial(
        adata,
        color=["DeepST_refine_domain"],  # Color by domain
        frameon=False,
        spot_size=150,
        title=f"Spatial Domains - {sample_name}"
    )
    
    # Save results
    # output_file = os.path.join(RESULTS_DIR, f"{SAMPLE_ID}_domains.pdf")
    # plt.savefig(output_file, bbox_inches="tight", dpi=300)
    ARI = metrics.adjusted_rand_score(adata.obs['DeepST_refine_domain'], adata.obs['ground_truth'])
    ARI_list.append(ARI)