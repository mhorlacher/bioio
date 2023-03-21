# %%
import numpy as np
import torch

# %%
def numpy_to_torch(x):
    if isinstance(x, str) or isinstance(x, bytes):
        if isinstance(x, str):
            x = x.encode('UTF-8')
        x = np.frombuffer(x, dtype=np.uint8)
    return torch.tensor(x)