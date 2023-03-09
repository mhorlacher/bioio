# %%
import importlib

import pandas as pd

# %%
def import_object_from_string(import_string):
    import_string = import_string.strip()
    module_name, object_name = '.'.join(import_string.split('.')[:-1]), import_string.split('.')[-1]
    module = importlib.import_module(module_name, object_name)
    return module.__getattribute__(object_name)

# %%
def flatten_dict(data_dict):
    return pd.json_normalize(data_dict, sep='/').to_dict(orient='records')[0]