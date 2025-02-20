# Shared Config class
class Config(object): 
    """
    A shared configuration class for managing device settings, data paths, 
    and training parameters in SemanticST.

    Attributes:
        use_cuda (bool): Whether to use CUDA (GPU) acceleration. Default is True.
        device (str): The computing device to be used ('cuda' by default). Can be set to 'cpu' if needed.
        spot_paths (list or None): Paths to ST dataset, if provided.
        seed (int): Random seed for reproducibility (default: 12345).
        dtype (torch.dtype or None): Specifies the data type for tensor operations.
        use_mini_batch (bool): Determines whether mini-batch processing is enabled.
        batch_size (int, optional): Defined only if `use_mini_batch=True`. Default is 3000, 
                                    specifying the number of samples per mini-batch.

    """
    def __init__(self, spot_paths=None,device='cuda',dtype=None, use_mini_batch=False):
        self.use_cuda = True
        self.device = device
        self.spot_paths = spot_paths 
        self.seed = 12345
        self.dtype = dtype 
        self.use_mini_batch=use_mini_batch
        # Specific settings for mini-batch training
        if use_mini_batch:
            self.batch_size = 3000  


 



