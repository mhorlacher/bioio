# %%
import tensorflow as tf
import pandas as pd

# %%
def flatten_dict(data_dict):
    return pd.json_normalize(data_dict, sep='/').to_dict(orient='records')[0]

# %%
base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# %%
baseComplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# %%
def reverse_complement(dna_string):
    """Returns the reverse-complement for a DNA string."""

    complement = [baseComplement.get(base, 'N') for base in dna_string]
    reversed_complement = reversed(complement)
    return ''.join(list(reversed_complement))

# %%
def sequence2int(sequence, mapping=base2int):
    return [mapping.get(base, 999) for base in sequence]

# %%
def sequence2onehot(sequence, mapping=base2int):
    return tf.one_hot(sequence2int(sequence, mapping), depth=4)