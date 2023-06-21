# %%
import tensorflow as tf

# %%
# Vocabulary for mapping characters (e.g. bases) to integers
sequence2int_mapping = {
    'DNA': tf.keras.layers.StringLookup(vocabulary=['A', 'C', 'G', 'T'], encoding=None, oov_token='N', num_oov_indices=1),
}

# %%
def sequence2int(sequence, vocab='DNA'):
    """
    Maps a sequence of characters to a sequence of integers.
    """
    
    return sequence2int_mapping[vocab](tf.strings.bytes_split(sequence)) - 1

# %%
def sequence2onehot(sequence, vocab='DNA'):
    """
    Maps a sequence of characters to a one-hot encoding.
    """

    return tf.one_hot(sequence2int(sequence, vocab), depth=len(sequence2int_mapping[vocab].input_vocabulary))