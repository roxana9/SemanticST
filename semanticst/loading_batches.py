#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:14:45 2024

@author: tower2
"""
import torch
import os
import random
from datetime import datetime
import numpy as np
import copy
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .preprocess import preprocess
from scipy.sparse import issparse
import torch.utils.data as data
import math

def Anndata_reader(file_name,dtype):
    """
    Reads spatial transcriptomics data from an AnnData file and applies preprocessing.

    Args:
        file_name (str): Path to the input file.
        dtype (str): Type of dataset ('Visium', 'Xenium', or generic H5AD).

    Returns:
        tuple: (data matrix, input dimension, sample count, x_pixel coordinates, y_pixel coordinates, AnnData object)

    """
    if dtype=="Visium":
        
        adata = sc.read_visium(file_name)
    elif dtype=="Xenium":
        
        adata=sc.read_h5ad(file_name)
        sc.pp.filter_cells(adata, min_counts=10)
        sc.pp.filter_genes(adata, min_cells=5)
       
    else:
        adata=sc.read_h5ad(file_name)
        
    adata.var_names_make_unique()
    preprocess(adata,dtype)
        
    if dtype!="Xenium":
        adata = adata[:, adata.var['highly_variable']]
   
        
    if issparse(adata.X):
        data = adata.X.A
    else:
       data = adata.X
#   pca.fit(data)
#   data = pca.transform(data) # get principal components
    x_pixel = np.array([adata.obsm['spatial'][:, 0].tolist()]).T
    y_pixel = np.array([adata.obsm['spatial'][:, 1].tolist()]).T
   
    return data, data.shape[1], data.shape[0], x_pixel, y_pixel,adata
def read_from_file(data_path,seed,dtype):
    """
   Reads spatial transcriptomics data and combines spatial coordinates with gene expression data.

   Args:
       data_path (str): Path to the dataset.
       seed (int): Random seed for reproducibility.
       dtype (str): Dataset type ('Visium', 'Xenium', or generic H5AD).

   Returns:
       tuple: (input size, number of samples, data matrix with spatial coordinates, AnnData object)

   """
    data = None
    input_size = 0
   
    if dtype=='Xenium':
        data, input_size, sample_num, x_pixel, y_pixel,adata = Anndata_reader(data_path,dtype)
    else:
        data, input_size, sample_num, x_pixel, y_pixel,adata = Anndata_reader(data_path,dtype)
    data = np.concatenate((data, x_pixel,y_pixel), 1)
   
    return input_size, sample_num,  data,adata
#data, input_size, sample_num, x_pixel, y_pixel = Anndata_reader(config_used.spot_paths)
#data = np.concatenate((data, x_pixel,y_pixel), 1)

class Dataloader(data.Dataset):
    """
    PyTorch dataset class for handling spatial transcriptomics data.

    Attributes:
        train (bool): Whether to load training data.
        input_size (int): Number of input features.
        sample_num (int): Number of samples in the dataset.
        data (numpy.ndarray): Data matrix including gene expression and spatial coordinates.
        adata (AnnData): Original AnnData object.

    Methods:
        __getitem__(index): Returns a single sample.
        __len__(): Returns dataset size.
    """
    def __init__(self, train=True, data_path=None,dtype=None, seed=2022):
        self.train = train
        self.input_size, self.sample_num, self.data,self.adata = read_from_file(
            data_path, seed,dtype)
       
    def __getitem__(self, index):
            """
        Retrieves a sample by index.

        Args:
            index (int): Index of the sample.

        Returns:
            numpy.ndarray: A single sample from the dataset.
        """
            sample = np.array(self.data[index])  
            return sample

    def __len__(self):
        return self.data.shape[0]
class PrepareDataloader():
    """
    Prepares PyTorch DataLoaders for training and testing spatial transcriptomics datasets.

    Attributes:
        config (Config): Configuration object.
        sample_num (int): Total number of samples.
        batch_size (int): Adjusted batch size based on dataset size.
        adata (AnnData): AnnData object containing dataset.
        train_loader (DataLoader): PyTorch DataLoader for training.
        test_loader (DataLoader): PyTorch DataLoader for testing.

    Methods:
        getloader(): Returns the training and test DataLoaders.
    """

    def __init__(self, config):
        self.config = config
       
            
        # hardware constraint
        kwargs = {'num_workers': 0, 'pin_memory': False} 

        self.sample_num = 0
        self.batch_size = 0

        trainset = Dataloader(True, config.spot_paths,config.dtype,  config.seed)
        self.sample_num += len(trainset)
        self.adata = trainset.adata
        if (self.sample_num % config.batch_size) < (0.1*config.batch_size):
            self.batch_size = config.batch_size + math.ceil((self.sample_num % config.batch_size)/(self.sample_num//config.batch_size))
        else:
            self.batch_size = config.batch_size

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=
            self.batch_size, shuffle=True, **kwargs) # mini-batches used for training are shuffled in this step.

        testset = Dataloader(False, config.spot_paths,config.dtype, config.seed)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=
            self.batch_size, shuffle=False, **kwargs) # mini-batches used for writing results are not shuffled.



        self.train_loader = train_loader
        self.test_loader = test_loader
        print("sample_num :", self.sample_num)
        print("batch_size :", self.batch_size)

    def getloader(self):
        """
        Returns the training and test data loaders.

        Returns:
            tuple: (train_loader, test_loader, number of batches, AnnData object)

        """
        return self.train_loader, self.test_loader,  math.ceil(self.sample_num/self.batch_size),self.adata
