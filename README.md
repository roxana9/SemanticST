# SemanticST

![SemanticST Pipeline](https://github.com/roxana9/SemanticST/raw/main/semanticst_figure.png)

## 🔷 SemanticST Overview

**SemanticST** is a graph neural network-based unsupervised deep learning approach for spatial transcriptomics data analysis. SemanticST employs a sophisticated learning strategy to integrate gene expression and spatial information, enabling the model to learn a latent representation of spatial transcriptomics (ST) data.
SemanticST uses a learnable weighted graph representation, termed Semantic Graphs, to better capture the complexity and diversity of biological processes.
For each semantic graph, a unique embedding is learned using an autoencoder with Graph Convolutional Network (GCN) layers, representing distinct semantic features in latent space. To combine these representations, SemanticST introduce a learnable weight, referred to as the semantic score, for each semantic graph. The final graph representation is then dynamically fused by weighting and combining the individual embeddings based on their semantic scores, resulting in a more accurate and adaptive graph representation.
SemanticST incorporate the mincut loss function together with the Deep MinCut (DMC) learning algorithm. This approach not only captures the global structure of the graph but also reduces redundant training time. More importantly, it ensures that the learned embeddings are both meaningful and interpretable, providing a more robust and insightful representation of the graph. 
Notably, we incorporated a mini-batch training option in SemanticST by training the model on spatial graphs in smaller batches, allowing the learned semantics graph to maintain both local and global perspectives across batches. This feature makes SemanticST memory-efficient and scalable, enabling its application to any spatial transcriptomics dataset, regardless of the number of samples.
---

## 🔷 Requirements  
To run **SemanticST**, install the following dependencies:

```bash
python==3.9.20
numpy==1.23.4
anndata==0.10.9
h5py==3.12.1
leidenalg==0.10.2
louvain==0.8.2
matplotlib==3.9.2
numba==0.60.0
scanpy==1.10.3
scikit-image==0.24.0
scikit-learn==1.5.2
scikit-misc==0.3.1
torch==2.5.1
torch-geometric==2.6.1
torch_scatter==2.1.2+pt25cu124
torch_sparse==0.6.18+pt25cu124
torchaudio==2.5.1
torchvision==0.20.1
torchviz==0.0.2
---

To install all dependencies, run:
'''bash
pip install -r requirements.txt
---

## 🚀 Installation  

You can install **SemanticST** via **PyPI**:
```bash
pip install semanticst
---
Or install from GitHub:
```bash
pip install git+https://github.com/roxana9/SemanticST.git
---

## 🔷 Step-by-Step Tutorial  

For a full tutorial on how to use **SemanticST**, visit our documentation:  

🔗 [**SemanticST Tutorial Webpage**](https://your-temp-webpage-link.com)  

_(The webpage is still under development, but a temporary version is available.)_
