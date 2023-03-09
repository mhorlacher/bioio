# %%
__version__ = '0.1.1'

# %%
import os
# This needs to be done *before* tensorflow is imported. 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
from bioio.dataspec import load_biospec