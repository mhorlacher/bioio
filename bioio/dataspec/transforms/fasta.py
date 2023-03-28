# %%
import pysam
import tensorflow as tf

from bioio.tf.utils import better_py_function_kwargs
from bioio.utils import sequence2onehot, reverse_complement

# %%
class Fasta():
    tensor_spec = tf.TensorSpec(shape=(None, 4), dtype=tf.int8)

    def __init__(self, filepath, to_onehot=True) -> None:
        self.to_onehot = to_onehot
        self._fasta = pysam.FastaFile(filepath)
    
    def fetch(self, chrom, start, end, strand='+', **kwargs):
        sequence = self._fasta.fetch(chrom, start, end).upper()

        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')
        
        if self.to_onehot:
            sequence = tf.cast(sequence2onehot(sequence), tf.int8)
            sequence.set_shape((None, 4))
        else:
            sequence = tf.convert_to_tensor(sequence)
            sequence.set_shape((None, ))
        return sequence

    def __call__(self, example):
        # return better_py_function_kwargs(Tout=tf.int8)(self.fetch)(kwargs)
        tensor = better_py_function_kwargs(Tout=self.tensor_spec)(self.fetch)(example)
        return tensor