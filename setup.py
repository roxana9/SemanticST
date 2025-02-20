from setuptools import setup, find_packages

setup(
    name="semanticst",  
    version="0.1.0",
    author="Roxana Zahedi",
    author_email="r.zahedi_nasab@unsw.edu.au",
    description="SemanticST: Spatially informed semantic graph learning for effective clustering and integration of spatial transcriptomics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/roxana9/SemanticST",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.4",
        "anndata==0.10.9",
        "h5py==3.12.1",
        "leidenalg==0.10.2",
        "louvain==0.8.2",
        "matplotlib==3.9.2",
        "numba==0.60.0",
        "scanpy==1.10.3",
        "scikit-image==0.24.0",
        "scikit-learn==1.5.2",
        "scikit-misc==0.3.1",
        "torch==2.5.1",
        "torch-geometric==2.6.1",
        "torch_scatter==2.1.2+pt25cu124",
        "torch_sparse==0.6.18+pt25cu124",
        "torchaudio==2.5.1",
        "torchvision==0.20.1",
        "torchviz==0.0.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
