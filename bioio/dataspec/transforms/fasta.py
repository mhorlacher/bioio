# %%
import pysam
import tensorflow as tf

from bioio.tf.utils import better_py_function_kwargs
from bioio.utils import sequence2onehot, reverse_complement, mask_noncanonical_bases

# %%
class Fasta():
    # tensor_spec = tf.TensorSpec(shape=(None, 4), dtype=tf.int8)

    def __init__(self, filepath, to_onehot=True, mask_noncanonical_bases=True) -> None:
        self.to_onehot = to_onehot
        self.mask_noncanonical_bases = mask_noncanonical_bases
        self._fasta = pysam.FastaFile(filepath)
        self.dtype = tf.int8 if to_onehot else tf.string
    
    def fetch(self, chrom, start, end, strand='+', **kwargs):
        sequence = self._fasta.fetch(chrom, start, end).upper()

        if self.mask_noncanonical_bases:
            # Everything except A, C, G, T will be mapped to N
            sequence = mask_noncanonical_bases(sequence)

        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')
        
        if self.to_onehot:
            sequence = tf.cast(sequence2onehot(sequence), self.dtype)
            # sequence.set_shape((None, 4))
        else:
            sequence = tf.convert_to_tensor(sequence)
            # sequence.set_shape(())
        return sequence

    def __call__(self, example):
        return better_py_function_kwargs(Tout=self.dtype)(self.fetch)(example)