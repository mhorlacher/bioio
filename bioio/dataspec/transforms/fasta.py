# %%
import pysam
import tensorflow as tf

from bioio.tf.ops import better_py_function_kwargs

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
class Fasta():
    tensor_spec = tf.TensorSpec(shape=(None, 4), dtype=tf.int8)

    def __init__(self, filepath) -> None:
        self._fasta = pysam.FastaFile(filepath)
    
    def fetch(self, chrom, start, end, strand='+', **kwargs):
        sequence = self._fasta.fetch(chrom, start, end).upper()

        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')
        
        return tf.cast(sequence2onehot(sequence), self.tensor_spec.dtype)

    def __call__(self, kwargs):
        # return better_py_function_kwargs(Tout=tf.int8)(self.fetch)(kwargs)
        tensor = better_py_function_kwargs(Tout=self.tensor_spec)(self.fetch)(kwargs)
        tensor.set_shape(self.tensor_spec.shape)
        return tensor