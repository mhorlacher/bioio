# %%
import copy
import math

import numpy as np
import tensorflow as tf

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

# %%
def sequences2inputs(sequences):
    """Converts one or more sequences to their batched one-hot representation.

    Args:
        sequences (str, list): Sequence or list of sequences

    Raises:
        ValueError: If a value not of type 'str' or 'list' is supplied. 

    Returns:
        tf.Tensor: Batched one-hot encoded sequence(s). 
    """
    
    if isinstance(sequences, str):
        sequences = [sequences]
    elif isinstance(sequences, list):
        pass
    else:
        raise ValueError(f'Unknown inputs type: {type(sequences)}')
    return tf.one_hot([sequence2int(seq) for seq in sequences], depth=4)

# %%
def nan_to_zero(x):
    """Replaces nan's with zeros."""
    return 0 if math.isnan(x) else x

# %%
def subset_nested_dict(d1, d2, _nest_level=0):
    """Subsets dict 1 (d1) by keys of dict 2 (d2). 

    Args:
        d1 (dict): Dictionary 1. 
        d2 (dict): Dictionary 2. 
        _nest_level (int, optional): Private argument used for better error message. Defaults to 0.

    Raises:
        ValueError: Raised if a (nested) key in d2 is not found in d1. 

    Returns:
        dict: Subsetted dictionary. 
    """
    
    subset_dict = dict()
    for key in d2:
        if key not in d1:
            raise ValueError(f"Missing key '{key}' at level {_nest_level}.")
        
        if isinstance(d2[key], dict):
            subset_dict[key] = subset_nested_dict(d1[key], d2[key], _nest_level=_nest_level+1)
        else:
            subset_dict[key] = d1[key]
    return subset_dict

# %%
def merge_nested_dicts(d1, d2, _nest_level=0):
    """Merges two dicts. 

    Args:
        d1 (dict): Dictionary 1. 
        d2 (dict): Dictionary 2. 
        _nest_level (int, optional): Private argument used for better error messages during recursion. Defaults to 0.

    Raises:
        ValueError: Raised if a duplicate atomic key is encountered at the same nesting level. 

    Returns:
        dict: Merged dictionary. 
    """
    
    d1 = copy.deepcopy(d1)
    for key in d2:
        if key not in d1:
            d1[key] = d2[key]
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            d1[key] = merge_nested_dicts(d1[key], d2[key], _nest_level=_nest_level+1)
        else:
            raise ValueError(f'Encountered duplicate atomic value {d1[key]} at depth {_nest_level}.')
    return d1

# %%
