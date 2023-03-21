# %%
import pysam
import tensorflow as tf

from bioio.tf.utils import better_py_function_kwargs
from bioio.utils import sequence2onehot, reverse_complement

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