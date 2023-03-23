# %%
import sys, math

import tensorflow as tf
import numpy as np
# import pyBigWig

from bioio.tf.utils import better_py_function_kwargs

# %%
def nan_to_zero(x):
    """Replaces nan's with zeros."""
    return 0 if math.isnan(x) else x

# %%
class BigWig():
    tensor_spec = tf.TensorSpec(shape=(None, ), dtype=tf.float32)

    def __init__(self, bigwig_filepath) -> None:
        try:
            import pyBigWig
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please install pyBigWig. See https://github.com/deeptools/pyBigWig')

        self._bigWig = pyBigWig.open(bigwig_filepath)
    
    def values(self, chrom, start, end, **kwargs):
        chrom, start, end = str(chrom), int(start), int(end) # not sure why this is needed, it worked locally with numpy.int32
        values = self._bigWig.values(chrom, start, end)
        # try:
        #     print(f'INFO: ({chrom},{start},{end})', file=sys.stderr)
        #     print(f'INFO: ({type(chrom)},{type(start)},{type(end)})', file=sys.stderr)
        #     values = self._bigWig.values(chrom, start, end)
        # except RuntimeError as e:
        #     # print(type(chrom), type(start), type(end))
        #     # raise e
        #     # TODO: Use logging module instead. This allows users to turn off warnings. 
        #     tf.print(start)
        #     print('WARNING: ' + str(e), file=sys.stderr)
        #     print(f'WARNING: ({chrom},{start},{end})', file=sys.stderr)
        #     print(f'WARNING: ({type(chrom)},{type(start)},{type(end)})', file=sys.stderr)
        #     return [0.0]*(end-start)
        
        values = [nan_to_zero(v) for v in values]        
        return tf.cast(values, self.tensor_spec.dtype)

    def __call__(self, kwargs):
        tensor = better_py_function_kwargs(Tout=self.tensor_spec)(self.values)(kwargs)
        tensor.set_shape(self.tensor_spec.shape)
        return tensor

# %%
class StrandedBigWig():
    tensor_spec = tf.TensorSpec(shape=(None, ), dtype=tf.float32)

    def __init__(self, bigwig_plus, bigwig_minus, reverse_minus=True) -> None:
        self._bigWig_plus = BigWig(bigwig_plus)
        self._bigWig_minus = BigWig(bigwig_minus)
        self.reverse_minus = reverse_minus

    def values(self, chrom, start, end, strand='+', **kwargs):
        """Returns values for a given range and strand. 
        
        Args:
            chrom  (str): Chromosome (chr1, chr2, ...)
            start  (int): 0-based start position
            end    (int): 0-based end position
            strand (str): Strand ('+' or '-')
            
        Returns:
            numpy.array: Numpy array of shape (end-start, )
        """

        if strand == '+':
            bigWig = self._bigWig_plus
        elif strand == '-':
            bigWig = self._bigWig_minus
        else:
            raise ValueError(f'Unexpected strand: {strand}')

        values = bigWig.values(chrom, start, end)

        if strand == '-' and self.reverse_minus:
            values = list(reversed(values))

        return tf.cast(values, dtype=self.tensor_spec.dtype)

    # def __call__(self, *args, **kwargs):
    #     return self.values(*args, **kwargs)

    def __call__(self, kwargs):
        tensor = better_py_function_kwargs(Tout=self.tensor_spec)(self.values)(kwargs)
        tensor.set_shape(self.tensor_spec.shape)
        return tensor
        