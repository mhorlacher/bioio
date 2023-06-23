# %%
__version__ = '0.1.1'

# %%
import os
# This needs to be done *before* tensorflow is imported. 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
# Disable absl INFO and WARNING log messages
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# %%
from . import tf, tfds #, torch
from bioio.dataspec import load_biospec