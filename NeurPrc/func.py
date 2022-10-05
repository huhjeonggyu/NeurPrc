import numpy as np
import torch 

def sqrt(x) :
    if   np.isscalar(x)             : return np.sqrt(x)
    elif isinstance(x,torch.Tensor) : return torch.sqrt(x)
    else                            : return torch.sqrt(x.data)

def exp(x) :
    if   np.isscalar(x)             : return np.exp(x)
    elif isinstance(x,torch.Tensor) : return torch.exp(x)
    else                            : return torch.exp(x.data)

def relu(x) :
    if   np.isscalar(x)             : return np.maximum(x)
    elif isinstance(x,torch.Tensor) : return torch.relu(x)
    else                            : return torch.relu(x.data)
